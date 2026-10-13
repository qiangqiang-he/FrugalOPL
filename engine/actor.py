"""The generic vLLM Actor worker.

The Actor owns generation mechanics only: vLLM, LoRA refresh, Redis I/O, and
the conversion of vLLM outputs into a CPU-only trajectory dictionary.  It
does not know how a training prompt is constructed.  For training, an
``AlgorithmPlugin`` receives raw Controller signals and returns already
chat-templated prompts together with sampling and return requirements.

Evaluation is intentionally separate.  It has no algorithm plugin dependency
and uses framework-owned evaluation message templates and generation
configuration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import multiprocessing as mp
import os
import pickle
import queue
import re
import signal
import string
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from algorithm_plugins import plugin_factory_from_config
from algorithm_plugins.algorithm_base import AlgorithmPlugin, Signals
from engine.gpu_gate import GpuGate, GpuGateRegistry
from engine.pipeline import resolve_training_pipeline
from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig,
    apply_generation_chat_template,
    controller_questions_and_answers,
    resolve_algorithm_runtime_config,
)


DEFAULT_RETURN_SIGNALS: dict[str, object] = {
    "responses": True,
    "response_ids": True,
    "prompt_ids": True,
    "prompt_indices": True,
    "response_truncated": True,
    "response_logprobs": True,
    "topk": {
        "enabled": False,
        "top_k": 0,
    },
}

# Evaluation needs completed answers, not the training-only logprob signals
# used for pi_old, KL, or top-k overlap.
EVALUATION_RETURN_SIGNALS: dict[str, object] = {
    **DEFAULT_RETURN_SIGNALS,
    "response_logprobs": False,
}

_TRAIN_REQUEST_FIELDS = frozenset({"prompts", "sampling", "return_signals", "require_return"})

# A vLLM EngineCore can take appreciably longer than a Redis polling interval
# to release its CUDA allocations.  In particular, that release happens while
# holding the physical-GPU gate when Actor shares a card with Learner or
# Teacher.  Give that cooperative path a real window before considering a
# forceful signal.  These are intentionally framework constants rather than
# per-run knobs: they protect process cleanup, not experiment semantics.
_EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC = 5.0
_SHARED_GPU_SHUTDOWN_GRACE_SEC = 30.0
_TERMINATE_SHUTDOWN_GRACE_SEC = 5.0
_KILL_SHUTDOWN_JOIN_SEC = 5.0


@dataclass
class _GenerateRequest:
    """One caller's generation request queued for a merged vLLM invocation."""

    prompts: list[str]
    sampling: object
    require_return: dict[str, object]
    future: Future
    sampling_params: list[object] = field(default_factory=list)


def load_yaml_config(path: str | Path) -> dict[str, object]:
    """Load one YAML configuration mapping without mutating it."""

    try:
        import yaml
    except ImportError as error:  # pragma: no cover - reported in runtime environments only.
        raise RuntimeError("PyYAML is required to load FrugalOPL configuration files.") from error

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML configuration must contain a mapping: {resolved}")
    return copy.deepcopy(dict(data))


def _resolve_extended_config_path(raw_path: object, current_config: Path) -> Path:
    """Resolve one ``extends`` entry relative to its declaring YAML file."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Top-level 'extends' must be a non-empty YAML path or a list of paths.")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = current_config.parent / candidate
    return candidate.resolve()


def load_yaml_config_tree(
    path: str | Path,
    ancestry: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Load a YAML mapping and recursively merge its optional ``extends`` bases.

    Configuration inheritance is deliberately a composition concern rather
    than a runtime setting: the returned mapping never contains ``extends``.
    Entries are resolved relative to the YAML file which declares them, and a
    cycle is rejected with its full path chain.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved in ancestry:
        chain = " -> ".join(str(item) for item in (*ancestry, resolved))
        raise ValueError(f"Cyclic YAML config inheritance: {chain}")

    overlay = load_yaml_config(resolved)
    extends = overlay.pop("extends", None)
    if extends is None:
        return overlay

    if isinstance(extends, str):
        base_entries: list[object] = [extends]
    elif isinstance(extends, Sequence) and not isinstance(extends, (bytes, bytearray)):
        base_entries = list(extends)
    else:
        raise ValueError("Top-level 'extends' must be a YAML path or a list of YAML paths.")
    if not base_entries:
        raise ValueError("Top-level 'extends' may not be an empty list.")

    merged_base: dict[str, object] = {}
    for entry in base_entries:
        base_path = _resolve_extended_config_path(entry, resolved)
        base = load_yaml_config_tree(base_path, (*ancestry, resolved))
        merged_base = merge_configs(merged_base, base)
    return merge_configs(merged_base, overlay)


def merge_configs(
    framework_config: Mapping[str, object], algorithm_config: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Recursively overlay an algorithm config on a framework config.

    A recursive merge lets an algorithm override its selected prompt alias,
    while preserving framework-owned defaults such as Redis and GPU settings.
    Lists are intentionally replaced rather than concatenated.
    """

    if not isinstance(framework_config, Mapping):
        raise TypeError("framework_config must be a mapping.")
    merged: dict[str, object] = copy.deepcopy(dict(framework_config))
    if algorithm_config is None:
        return merged
    if not isinstance(algorithm_config, Mapping):
        raise TypeError("algorithm_config must be a mapping when provided.")

    def overlay(target: dict[str, object], additions: Mapping[str, object]) -> None:
        for key, value in additions.items():
            current = target.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested = copy.deepcopy(dict(current))
                overlay(nested, value)
                target[str(key)] = nested
            else:
                target[str(key)] = copy.deepcopy(value)

    overlay(merged, algorithm_config)
    return merged


def load_merged_config(
    framework_config_path: str | Path,
    algorithm_config_path: str | Path | None = None,
) -> dict[str, object]:
    """Load the framework config and, optionally, its algorithm overlay."""

    framework = load_yaml_config_tree(framework_config_path)
    algorithm = load_yaml_config_tree(algorithm_config_path) if algorithm_config_path is not None else None
    return merge_configs(framework, algorithm)


def gpu_memory_headroom_is_safe(
    *,
    free_bytes: int,
    total_bytes: int,
    gpu_memory_utilization: float,
    minimum_free_gb: float,
) -> bool:
    """Return whether starting vLLM still leaves the required VRAM floor.

    vLLM reserves roughly ``total_memory * gpu_memory_utilization``.  The
    check is deliberately conservative: it uses *currently* free memory, not
    merely total device memory, so sharing a local GPU with the Learner cannot
    silently consume the project's mandatory 3 GB safety margin.
    """

    if total_bytes <= 0 or free_bytes < 0:
        raise ValueError("GPU memory values must be non-negative with a positive total.")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("actor.engine.gpu_memory_utilization must be strictly between 0 and 1.")
    if minimum_free_gb < 0:
        raise ValueError("minimum_free_gpu_memory_gb must be non-negative.")
    minimum_free_bytes = int(minimum_free_gb * 1024**3)
    planned_reservation = math.ceil(total_bytes * gpu_memory_utilization)
    return free_bytes - planned_reservation >= minimum_free_bytes


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _torch() -> Any:
    """Import torch only in a worker path that actually needs tensors/CUDA."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - configuration/runtime failure.
        raise RuntimeError("PyTorch is required by the Actor worker.") from error
    return torch


def _destroy_vllm_worker_process_groups(_worker: object) -> None:
    """Destroy vLLM's EngineCore distributed state inside its worker process.

    A single-GPU vLLM ``EngineCore`` still initializes a NCCL process group.
    In the vLLM version used by this project, the normal single-process
    executor shutdown releases the worker but does not destroy that group.
    This module-level function is deliberately serializable with
    ``cloudpickle`` and is sent as bytes through ``LLM.collective_rpc``;
    vLLM's utility RPC then executes it in the EngineCore worker itself.

    ``_worker`` is the vLLM WorkerWrapper argument required by that RPC
    protocol.  The cleanup APIs are process-global, so it is not otherwise
    needed here.
    """

    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    # Keep the same model-parallel-then-process-group ordering used by
    # vLLM's multi-process executor.  ``finally`` makes the base distributed
    # state cleanup best-effort even if a partially initialized model-parallel
    # structure reports an error.
    try:
        destroy_model_parallel()
    finally:
        destroy_distributed_environment()


def _gpu_memory_info(gpu_id: int) -> tuple[int, int]:
    """Read *device-wide* free/total VRAM without initializing CUDA locally.

    vLLM owns a separate engine process.  ``torch.cuda.mem_get_info()`` in the
    parent can be process-context dependent in WSL, so it is not strong enough
    for a hard global safety floor.  ``nvidia-smi`` reports device-wide values
    and also lets us check memory before importing CUDA or vLLM.
    """

    command = [
        "nvidia-smi",
        "--id",
        str(gpu_id),
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:  # pragma: no cover - host setup failure.
        raise RuntimeError(
            "nvidia-smi is required to enforce FrugalOPL's device-wide 3 GB GPU safety floor."
        ) from error
    except subprocess.CalledProcessError as error:  # pragma: no cover - host setup failure.
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"Could not query GPU {gpu_id} memory with nvidia-smi: {detail}") from error

    line = completed.stdout.strip().splitlines()
    if len(line) != 1:
        raise RuntimeError(f"Unexpected nvidia-smi memory output for GPU {gpu_id}: {completed.stdout!r}")
    fields = [field.strip() for field in line[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"Unexpected nvidia-smi memory fields for GPU {gpu_id}: {line[0]!r}")
    try:
        free_mib, total_mib = (int(field) for field in fields)
    except ValueError as error:
        raise RuntimeError(f"Unexpected nvidia-smi memory values for GPU {gpu_id}: {line[0]!r}") from error
    return free_mib * 1024**2, total_mib * 1024**2


def _deep_cpu_detach(value: object) -> object:
    """Make tensors safe for Redis without changing ordinary signal values."""

    # Avoid importing torch merely to serialize a plain dictionary in tests.
    try:
        import torch
    except ImportError:  # pragma: no cover - Actor itself requires torch in production.
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _deep_cpu_detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_cpu_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_cpu_detach(item) for item in value)
    return value


def _decode_token_ids(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("Actor tokenizer must provide decode() to recover response text.")
    return decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _decode_generated_response(tokenizer: object | None, token_ids: list[int], fallback_text: str) -> str:
    if tokenizer is None:
        return fallback_text
    try:
        return _decode_token_ids(tokenizer, token_ids)
    except Exception:
        return fallback_text


def _response_was_truncated(response_output: object, token_count: int, max_tokens: int) -> bool:
    finish_reason = getattr(response_output, "finish_reason", None)
    if isinstance(finish_reason, str):
        reason = finish_reason.strip().lower()
        if reason in {"length", "max_tokens"}:
            return True
        if reason in {"stop", "eos"}:
            return False

    stop_reason = getattr(response_output, "stop_reason", None)
    if isinstance(stop_reason, str) and stop_reason.strip().lower() in {"length", "max_tokens"}:
        return True
    return max_tokens > 0 and token_count >= max_tokens


def _logprob_value(info: object) -> float:
    value = getattr(info, "logprob", None)
    return float("-inf") if value is None else float(value)


def _rank_value(info: object, fallback: int) -> int:
    rank = getattr(info, "rank", None)
    if rank is None:
        rank = getattr(info, "decoded_token_rank", None)
    return fallback if rank is None else int(rank)


def _topk_items(step_logprobs: object, top_k: int) -> tuple[list[int], list[float]]:
    if not step_logprobs:
        return [-1] * top_k, [float("-inf")] * top_k
    if not isinstance(step_logprobs, Mapping):
        raise TypeError("vLLM response logprobs must be mappings when present.")

    ranked: list[tuple[int, int, float]] = []
    for fallback, (token_id, info) in enumerate(step_logprobs.items()):
        ranked.append((_rank_value(info, fallback), int(token_id), _logprob_value(info)))
    ranked.sort(key=lambda item: item[0])
    token_ids = [token_id for _, token_id, _ in ranked[:top_k]]
    values = [value for _, _, value in ranked[:top_k]]
    if len(token_ids) < top_k:
        missing = top_k - len(token_ids)
        token_ids.extend([-1] * missing)
        values.extend([float("-inf")] * missing)
    return token_ids, values


def _response_logprobs(step_logprobs: object, token_ids: list[int]) -> object:
    values: list[float] = []
    for index, token_id in enumerate(token_ids):
        step = step_logprobs[index] if step_logprobs and index < len(step_logprobs) else None
        if isinstance(step, Mapping) and token_id in step:
            values.append(_logprob_value(step[token_id]))
        else:
            # vLLM can omit an entry on malformed/legacy responses.  Keeping
            # length aligned with response_ids is more useful than dropping it.
            values.append(0.0)
    return _torch().tensor(values, dtype=_torch().float32)


def _run_name_and_redis(config: Mapping[str, object]) -> tuple[str, str]:
    run = _mapping(config.get("run"), "run")
    name = run.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("run.name must be a non-empty string.")
    redis_cfg = _mapping(run.get("redis"), "run.redis")
    prefix = redis_cfg.get("key_prefix", "frugalopl")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("run.redis.key_prefix must be a non-empty string.")
    return name.strip(), prefix.strip()


def _lora_adapter_root(config: Mapping[str, object]) -> Path:
    run = _mapping(config.get("run"), "run")
    storage = _mapping(run.get("storage", {}), "run.storage")
    configured = storage.get("lora_adapter_root")
    if configured is not None:
        return Path(str(configured)).expanduser()
    output_root = Path(str(storage.get("output_root", "outputs"))).expanduser()
    name, _prefix = _run_name_and_redis(config)
    return output_root / name / "tmp" / "_tmp_lora"


def _latest_lora_adapter(config: Mapping[str, object]) -> tuple[int, str]:
    """Find the newest fully published adapter without creating directories."""

    root = _lora_adapter_root(config)
    patterns = (
        re.compile(r"^adapters_step_(\d+)$"),
        re.compile(r"^adapter_step_(\d+)$"),
        re.compile(r"^adapters_v(\d+)$"),
    )
    best_step = -1
    best_path = ""

    if not root.is_dir():
        return best_step, best_path
    for path in root.iterdir():
        if not path.is_dir() or path.name.endswith("_tmp"):
            continue
        match = re.match(r"^step_(\d+)$", path.name)
        if match:
            candidate = path / "adapter_lora"
            if candidate.is_dir() and int(match.group(1)) > best_step:
                best_step = int(match.group(1))
                best_path = str(candidate)
            continue
        for pattern in patterns:
            match = pattern.match(path.name)
            if match and int(match.group(1)) > best_step:
                best_step = int(match.group(1))
                best_path = str(path)
                break
    return best_step, best_path


class Actor:
    """One independent Actor process and its vLLM engine.

    ``gpu_id`` is the physical ID used by :class:`ActorManager`.  The child
    process masks it through ``CUDA_VISIBLE_DEVICES`` before this class imports
    CUDA libraries, so all runtime CUDA calls use logical device zero.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        plugin: AlgorithmPlugin | None = None,
        gpu_id: int | None = None,
        gpu_gate: GpuGate | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        self.config: dict[str, object] = copy.deepcopy(dict(config))
        self.run_cfg = _mapping(self.config.get("run"), "run")
        self.actor_cfg = _mapping(self.config.get("actor"), "actor")
        self.engine_cfg = _mapping(self.actor_cfg.get("engine"), "actor.engine")
        self.policy_cfg = _mapping(self.run_cfg.get("policy"), "run.policy")
        self.runtime_cfg = _mapping(self.run_cfg.get("runtime", {}), "run.runtime")
        self.lora_cfg = _mapping(self.actor_cfg.get("lora", {}), "actor.lora")
        # Training configurations own chat-template thinking mode in their
        # algorithm runtime.  Resolve that framework-owned contract once so
        # periodic Actor evaluation uses exactly the same value as the
        # Worker-local algorithm plugin.  A standalone Actor evaluation may
        # intentionally omit ``algorithm`` and retain its legacy local
        # compatibility setting below.
        configured_algorithm = self.config.get("algorithm")
        self._algorithm_runtime: AlgorithmRuntimeConfig | None = (
            resolve_algorithm_runtime_config(self.config)
            if configured_algorithm is not None
            else None
        )

        configured_gpu_ids = self.actor_cfg.get("gpu_ids", [])
        if not isinstance(configured_gpu_ids, list):
            raise ValueError("actor.gpu_ids must be a list of GPU ids.")
        if gpu_id is None:
            gpu_id = int(configured_gpu_ids[0]) if configured_gpu_ids else 0
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer.")
        if configured_gpu_ids and gpu_id not in configured_gpu_ids:
            raise ValueError("gpu_id must be one of actor.gpu_ids.")
        self.gpu_id = gpu_id
        if gpu_gate is not None and gpu_gate.physical_gpu_id != gpu_id:
            raise ValueError("Actor GPU gate must match the Actor physical gpu_id.")
        # Direct Actor construction in CPU tests and standalone use remains a
        # dedicated-device path.  Controller-managed Workers receive the
        # process-shared gate selected for this physical GPU.
        self.gpu_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=gpu_id)
        self.plugin = plugin

        self.max_batches_per_turn = int(self.engine_cfg.get("max_batches_per_turn", 1))
        self.collect_timeout_sec = float(self.engine_cfg.get("collect_timeout_sec", 0.03))
        if self.max_batches_per_turn < 1:
            raise ValueError("actor.engine.max_batches_per_turn must be at least 1.")
        if self.collect_timeout_sec < 0:
            raise ValueError("actor.engine.collect_timeout_sec must be non-negative.")

        logprobs_cfg = _mapping(self.engine_cfg.get("logprobs", {}), "actor.engine.logprobs")
        self.default_logprobs = logprobs_cfg.get("default", 0)
        self.max_logprobs = int(logprobs_cfg.get("max", 16))
        if self.default_logprobs is not None:
            self.default_logprobs = int(self.default_logprobs)
            if self.default_logprobs < 0:
                raise ValueError("actor.engine.logprobs.default must be non-negative or null.")
        if self.max_logprobs < 0:
            raise ValueError("actor.engine.logprobs.max must be non-negative.")
        if isinstance(self.default_logprobs, int) and self.default_logprobs > self.max_logprobs:
            raise ValueError("actor.engine.logprobs.default may not exceed actor.engine.logprobs.max.")

        policy_lora = _mapping(self.policy_cfg.get("lora", {}), "run.policy.lora")
        # A training Actor discovers newly published adapters under
        # ``run.storage.lora_adapter_root``.  Offline evaluation instead has
        # a deliberately explicit adapter choice: setting ``actor.lora.path``
        # to null or an empty string must mean *base model*, even when the
        # inherited training configuration has LoRA enabled.
        self._has_explicit_lora_path = "path" in self.lora_cfg
        self._explicit_lora_path = self._normalize_explicit_lora_path()
        if self._has_explicit_lora_path:
            self.lora_enabled = bool(self._explicit_lora_path)
        else:
            self.lora_enabled = bool(self.lora_cfg.get("enabled", False)) and bool(
                policy_lora.get("enabled", False)
            )
        self._policy_lora_rank = int(policy_lora.get("rank", 64))
        self._max_lora_rank = int(self.lora_cfg.get("max_rank", self._policy_lora_rank or 64))
        if self.lora_enabled and self._max_lora_rank < self._policy_lora_rank:
            raise ValueError("actor.lora.max_rank must be at least run.policy.lora.rank.")

        self.llm: object | None = None
        self.tokenizer: object | None = None
        self.current_lora_step = -1
        self.current_lora_path = ""
        self.current_lora_request: object | None = None
        self._failed_lora_paths: set[str] = set()

        # Tests may substitute this with a tiny CPU-only class.  Production
        # resolves vLLM's SamplingParams lazily, after CUDA visibility is set.
        self._sampling_params_factory: Callable[..., object] | None = None
        self._request_queue: queue.Queue[_GenerateRequest | None] = queue.Queue()
        self._generation_thread: threading.Thread | None = None
        self._generation_stop = threading.Event()
        self._thread_state = threading.local()
        self._metrics_lock = threading.Lock()
        self._completed_rollout_batches = 0
        self._cumulative_duration_sec = 0.0
        self._cumulative_response_tokens = 0
        self._cumulative_rollouts = 0
        self._cumulative_throughput = 0.0

    def _normalize_explicit_lora_path(self) -> str:
        """Resolve an explicit evaluation adapter, or select the base model.

        Presence of ``actor.lora.path`` is intentional configuration, so an
        unavailable directory is not treated as a reason to fall back to the
        training adapter scan.  It instead selects the unadapted policy.
        """

        configured = self.lora_cfg.get("path")
        if configured is None:
            return ""
        if not isinstance(configured, (str, os.PathLike)):
            raise TypeError("actor.lora.path must be a string, path-like value, null, or an empty string.")
        text = os.fspath(configured).strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if not path.is_dir():
            print(
                f"[Actor GPU {self.gpu_id}] Explicit LoRA adapter is unavailable; "
                f"using the base model: {path}",
                flush=True,
            )
            return ""
        return str(path)

    def set_plugin(self, plugin: AlgorithmPlugin) -> None:
        """Attach the worker-local algorithm plugin after tokenizer creation."""

        if not isinstance(plugin, AlgorithmPlugin):
            raise TypeError("plugin must inherit AlgorithmPlugin.")
        self.plugin = plugin

    def queue_keys(self) -> dict[str, str]:
        """Return all fixed Redis inbox/result keys for this run."""

        name, prefix = _run_name_and_redis(self.config)
        root = f"{prefix}:{name}"
        return {
            "actor_train": f"{root}:actor:train",
            "actor_eval": f"{root}:actor:eval",
            "teacher_train": f"{root}:teacher:train",
            "learner_train": f"{root}:learner:train",
            "controller_train_results": f"{root}:controller:train_results",
            "controller_eval_results": f"{root}:controller:eval_results",
            "controller_events": f"{root}:controller:events",
        }

    def route_output(self, task_type: str) -> str:
        """Choose a separate downstream Redis key for train versus evaluation."""

        keys = self.queue_keys()
        if task_type == "eval":
            return keys["controller_eval_results"]
        if task_type != "train":
            raise ValueError("task_type must be 'train' or 'eval'.")
        downstream_node = resolve_training_pipeline(self.config).successor_of("actor")
        name, prefix = _run_name_and_redis(self.config)
        return f"{prefix}:{name}:{downstream_node}:train"

    def initialize_engine(self) -> None:
        """Create vLLM only after verifying the 3 GB free-memory safety floor."""

        self._validate_engine_config()
        self._validate_memory_before_engine_creation()
        model_path = os.path.expanduser(str(self.policy_cfg.get("model_name_or_path", "")))
        if not model_path:
            raise ValueError("run.policy.model_name_or_path must be configured.")

        # These environment values must be set before the first vLLM import.
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
        os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "0")
        # In the vLLM version used here, CUDA Graph pools are not included in
        # the KV-cache budget unless this flag is enabled before import.  Graph
        # capture is the default, so keep the configured envelope meaningful.
        if not bool(self.engine_cfg.get("enforce_eager", False)) and bool(
            self.engine_cfg.get("estimate_cudagraph_memory", True)
        ):
            os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
        try:
            from vllm import LLM, SamplingParams

            self._sampling_params_factory = SamplingParams
            llm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": 1,
                "enable_lora": self.lora_enabled,
                "max_lora_rank": self._max_lora_rank,
                "max_logprobs": self.max_logprobs,
                "gpu_memory_utilization": float(self.engine_cfg.get("gpu_memory_utilization", 0.30)),
                "max_model_len": int(self.engine_cfg.get("max_model_len", 20480)),
                "dtype": str(self.policy_cfg.get("dtype", "bfloat16")),
                "trust_remote_code": True,
                "enforce_eager": bool(self.engine_cfg.get("enforce_eager", False)),
            }
            try:
                self.llm = LLM(**llm_kwargs)
            except Exception as error:
                # An explicit evaluation adapter is optional.  If vLLM cannot
                # even initialize its LoRA support, continue with exactly the
                # requested base model instead of failing the whole eval run.
                if not (self._has_explicit_lora_path and self.lora_enabled):
                    raise
                failed_path = self._explicit_lora_path
                print(
                    f"[Actor GPU {self.gpu_id}] Could not initialize LoRA; using the base model: "
                    f"{failed_path} ({error})",
                    flush=True,
                )
                self._mark_lora_path_failed(failed_path)
                self.lora_enabled = False
                llm_kwargs["enable_lora"] = False
                self.llm = LLM(**llm_kwargs)
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except Exception:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._validate_memory_after_engine_creation()
        except Exception:
            self._release_engine()
            raise

    def _validate_engine_config(self) -> None:
        if int(self.engine_cfg.get("tensor_parallel_size", 1)) != 1:
            raise ValueError(
                "actor.engine.tensor_parallel_size must be 1: Actor scales through one "
                "independent vLLM process per actor.gpu_ids entry."
            )
        utilization = float(self.engine_cfg.get("gpu_memory_utilization", 0.30))
        if not 0 < utilization < 1:
            raise ValueError("actor.engine.gpu_memory_utilization must be strictly between 0 and 1.")
        if int(self.engine_cfg.get("max_model_len", 0)) <= 0:
            raise ValueError("actor.engine.max_model_len must be positive.")
        if float(self.runtime_cfg.get("minimum_free_gpu_memory_gb", 3)) < 0:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb must be non-negative.")

    def _minimum_free_gpu_memory_gb(self) -> float:
        minimum = float(self.runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        if minimum < 0:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb must be non-negative.")
        return minimum

    def _validate_memory_before_engine_creation(self) -> None:
        free_bytes, total_bytes = _gpu_memory_info(self.gpu_id)
        utilization = float(self.engine_cfg.get("gpu_memory_utilization", 0.30))
        minimum = self._minimum_free_gpu_memory_gb()
        if not gpu_memory_headroom_is_safe(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            gpu_memory_utilization=utilization,
            minimum_free_gb=minimum,
        ):
            planned = total_bytes * utilization / 1024**3
            raise MemoryError(
                f"Refusing to initialize Actor on GPU {self.gpu_id}: {free_bytes / 1024**3:.2f} GB "
                f"is currently free, while vLLM may reserve {planned:.2f} GB and must leave "
                f"at least {minimum:.2f} GB free."
            )

    def _validate_memory_after_engine_creation(self) -> None:
        free_bytes, _total_bytes = _gpu_memory_info(self.gpu_id)
        minimum = self._minimum_free_gpu_memory_gb()
        if free_bytes < int(minimum * 1024**3):
            raise MemoryError(
                f"Actor initialization left only {free_bytes / 1024**3:.2f} GB free on GPU "
                f"{self.gpu_id}; the required safety floor is {minimum:.2f} GB."
            )

    def start_generation_loop(self) -> None:
        if self._generation_thread is not None and self._generation_thread.is_alive():
            return
        self._generation_stop.clear()
        self._generation_thread = threading.Thread(
            target=self._generation_loop,
            name=f"ActorGeneration-{self.gpu_id}",
            daemon=True,
        )
        self._generation_thread.start()

    def stop_generation_loop(self) -> bool:
        """Stop the request thread and report whether it fully quiesced.

        EngineCore utility RPCs must never race an in-flight ``generate``
        call from this thread.  A normal Controller shutdown waits for the
        synchronous generation turn to finish, but retain an explicit guard
        for an unexpectedly stalled native call.
        """

        self._generation_stop.set()
        self._request_queue.put(None)
        thread = self._generation_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                print(
                    f"[Actor GPU {self.gpu_id}] Generation thread did not stop within 5s; "
                    "skipping explicit EngineCore cleanup to avoid racing an in-flight request.",
                    flush=True,
                )
                return False
        self._generation_thread = None
        return True

    def generate_rollouts(
        self,
        prompts: list[str],
        sampling: object,
        require_return: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Generate trajectories from already-rendered prompts.

        The returned values are detached CPU tensors, strings, and lists, so
        they can be sent directly through Redis.  This method never renders a
        prompt, computes an algorithm signal, or accesses the Controller.
        """

        if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("prompts must be a list[str].")
        self._validate_sampling_input(sampling)
        # Validate requests before a caller waits on the generation thread.
        normalized_requirements = self._normalize_return_signals(require_return)
        self._required_vllm_logprobs(normalized_requirements)
        if not prompts:
            empty = self._empty_trajectory(normalized_requirements)
            self._thread_state.rollout_metadata = self._rollout_metadata(0.0, 0, 0, 0, 0)
            return empty
        if self.llm is None:
            raise RuntimeError("Actor engine is not initialized.")
        if self._generation_thread is None or not self._generation_thread.is_alive():
            raise RuntimeError("Actor generation loop is not running.")

        future: Future = Future()
        self._request_queue.put(
            _GenerateRequest(
                prompts=list(prompts),
                sampling=sampling,
                require_return=normalized_requirements,
                future=future,
            )
        )
        trajectory, metadata = future.result()
        self._thread_state.rollout_metadata = metadata
        return trajectory

    @staticmethod
    def _validate_sampling_input(sampling: object) -> None:
        if isinstance(sampling, Mapping):
            return
        if isinstance(sampling, list):
            if all(isinstance(item, Mapping) or hasattr(item, "logprobs") for item in sampling):
                return
            raise TypeError("sampling must be a mapping, SamplingParams, or a list of either.")
        if hasattr(sampling, "logprobs"):
            return
        raise TypeError("sampling must be a mapping, SamplingParams, or a list of either.")

    def pop_rollout_metadata(self) -> dict[str, object]:
        metadata = getattr(self._thread_state, "rollout_metadata", None)
        self._thread_state.rollout_metadata = None
        return dict(metadata or {})

    def process_training_batch(self, signals: Signals) -> Signals:
        """Run the Actor-side algorithm hook and attach its generic trajectory.

        A plugin has two supported equivalent request forms:

        ``{"prompts", "sampling", "return_signals"}`` (the compact form),
        or ``{"rollout_request": {"prompts", "sampling", "require_return"}}``.
        The first is the current canonical form.  Both preserve the important
        boundary that only the plugin creates training prompts.
        """

        source = self._validate_controller_signals(signals)
        if self.plugin is None:
            raise RuntimeError("Training batches require an algorithm plugin in the Actor worker.")
        hook_name = resolve_training_pipeline(self.config).node("actor").plugin_hook
        hook = getattr(self.plugin, hook_name or "", None)
        if not callable(hook):
            raise RuntimeError(f"Actor plugin hook is unavailable: {hook_name!r}.")
        produced = hook(copy.deepcopy(source))
        if not isinstance(produced, Mapping):
            raise TypeError(f"Actor plugin hook {hook_name!r} must return a mapping.")
        additions = dict(produced)
        if any(key in additions for key in ("questions", "answers", "records")):
            raise ValueError(f"Actor plugin hook {hook_name!r} may not replace Controller questions, answers, or records.")
        if "traj" in additions:
            raise ValueError(f"Actor plugin hook {hook_name!r} may not create 'traj'; Actor owns generic rollouts.")

        prompts, sampling, require_return = self._take_training_rollout_request(additions)
        trajectory = self.generate_rollouts(prompts, sampling, require_return)
        result: Signals = dict(source)
        result.update(additions)
        result["traj"] = trajectory
        metadata = self.pop_rollout_metadata()
        if metadata:
            result["actor_metrics"] = metadata
        return result

    def _take_training_rollout_request(
        self, additions: dict[str, object]
    ) -> tuple[list[str], object, Mapping[str, object] | None]:
        nested = additions.pop("rollout_request", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError("Actor plugin_hook's 'rollout_request' must be a mapping.")
            if any(field in additions for field in _TRAIN_REQUEST_FIELDS):
                raise ValueError("Use either rollout_request or top-level rollout fields, not both.")
            request = dict(nested)
        else:
            request = {}
            for field in _TRAIN_REQUEST_FIELDS:
                if field in additions:
                    request[field] = additions.pop(field)

        prompts = request.get("prompts")
        if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Actor plugin must return prompts as a list[str].")
        if "sampling" not in request:
            raise ValueError("Actor plugin must return sampling settings for training generation.")
        if "return_signals" in request and "require_return" in request:
            raise ValueError("Use only return_signals or require_return in an Actor rollout request.")
        require_return = request.get("return_signals", request.get("require_return"))
        if require_return is not None and not isinstance(require_return, Mapping):
            raise TypeError("Actor return_signals must be a mapping when provided.")
        return list(prompts), request["sampling"], require_return

    def process_evaluation_batch(self, signals: Signals) -> Signals:
        """Compatibility entry point for Redis-delivered evaluation batches.

        Controller preserves complete dataset records so templates can use
        any original field. Older callers without ``records`` retain the
        question/answer compatibility path.
        """

        source = self._validate_controller_signals(signals)
        records = source.get("records")
        if records is None:
            records = [
                {"question": question, "answer": answer}
                for question, answer in zip(source["questions"], source["answers"])
            ]
        generated = self.generate_evaluation_rollouts(records)
        result: Signals = dict(source)
        result["traj"] = generated["traj"]
        if "actor_metrics" in generated:
            result["actor_metrics"] = generated["actor_metrics"]
        return result

    def generate_evaluation_rollouts(
        self,
        records: list[Mapping[str, object]],
        *,
        prompt_alias: str | None = None,
    ) -> dict[str, object]:
        """Generate evaluation trajectories from complete dataset records.

        ``records`` stay in their input order, and the returned trajectory's
        flattened responses use its ``prompt_indices`` field to point back to
        those records.  Every named format field in the selected message
        template is filled from the record with the same key.  This keeps
        evaluation prompt flexibility at the data boundary without involving
        an algorithm plugin or Controller.
        """

        source = self._validate_evaluation_records(records)
        evaluation = _mapping(self.actor_cfg.get("evaluation", {}), "actor.evaluation")
        if not bool(evaluation.get("enabled", True)):
            raise RuntimeError("Actor evaluation is disabled by actor.evaluation.enabled.")
        if self.tokenizer is None:
            raise RuntimeError("Actor tokenizer is not initialized for evaluation prompt rendering.")

        generation = _mapping(evaluation.get("generation"), "actor.evaluation.generation")
        rollouts_per_prompt = int(evaluation.get("rollouts_per_prompt", 8))
        rollouts_per_batch = int(evaluation.get("rollouts_per_batch", 64))
        if rollouts_per_prompt < 1 or rollouts_per_batch < 1:
            raise ValueError("actor.evaluation rollout counts must be positive.")
        if rollouts_per_batch > 64:
            raise ValueError("actor.evaluation.rollouts_per_batch may not exceed the local evaluation limit of 64.")
        if len(source) * rollouts_per_prompt > rollouts_per_batch:
            raise ValueError(
                "An Actor evaluation batch may not exceed actor.evaluation.rollouts_per_batch "
                f"({rollouts_per_batch} rollouts)."
            )

        template = self._evaluation_template(prompt_alias)
        enable_thinking = self._evaluation_enable_thinking(evaluation)
        prompts = [
            self._apply_chat_template(
                self._render_evaluation_messages(template, record, record_index),
                enable_thinking=enable_thinking,
            )
            for record_index, record in enumerate(source)
        ]
        sampling = copy.deepcopy(generation)
        sampling["n"] = rollouts_per_prompt
        trajectory = self.generate_rollouts(prompts, sampling, EVALUATION_RETURN_SIGNALS)
        result: dict[str, object] = {"records": source, "traj": trajectory}
        metadata = self.pop_rollout_metadata()
        if metadata:
            result["actor_metrics"] = metadata
        return result

    def _evaluation_enable_thinking(self, evaluation: Mapping[str, object]) -> bool:
        """Resolve the chat-template mode without duplicating training config.

        A training run owns this mode at ``algorithm.enable_thinking``.  Both
        its algorithm plugin and its periodic Actor evaluation consume that
        same setting, so their prompts cannot silently diverge.  Standalone
        Actor-only evaluation deliberately has no ``algorithm`` section or
        plugin; it retains ``actor.evaluation.enable_thinking`` as its
        self-contained compatibility setting.
        """

        if self._algorithm_runtime is not None:
            return self._algorithm_runtime.enable_thinking

        value = evaluation.get("enable_thinking", False)
        if not isinstance(value, bool):
            raise TypeError("actor.evaluation.enable_thinking must be a boolean.")
        return value

    @staticmethod
    def _validate_evaluation_records(records: object) -> list[dict[str, object]]:
        if not isinstance(records, list):
            raise TypeError("Evaluation records must be a list of mappings.")
        normalized: list[dict[str, object]] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"Evaluation record {record_index} must be a mapping.")
            copied = copy.deepcopy(dict(record))
            question = copied.get("question")
            if not isinstance(question, str) or not question.strip():
                raise TypeError(f"Evaluation record {record_index} must contain a non-empty string 'question'.")
            if "answer" not in copied or copied["answer"] is None:
                raise ValueError(f"Evaluation record {record_index} must contain a non-null 'answer'.")
            normalized.append(copied)
        return normalized

    def _evaluation_template(
        self, prompt_alias: str | None = None
    ) -> str | tuple[tuple[str, str], ...]:
        from utils import prompts as prompt_templates

        configured = _mapping(self.run_cfg.get("prompts"), "run.prompts")
        alias = "evaluation" if prompt_alias is None else prompt_alias
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("prompt_alias must be a non-empty run.prompts alias when provided.")
        name = configured.get(alias)
        if not isinstance(name, str) or not name:
            raise ValueError(f"run.prompts.{alias} must name a prompt template.")
        template = getattr(prompt_templates, name, None)
        if name not in prompt_templates.__all__:
            raise ValueError(
                f"run.prompts.{alias} must name a prompt definition exported by utils.prompts; "
                f"got {name!r}."
            )
        if isinstance(template, str):
            return template
        if (
            isinstance(template, tuple)
            and template
            and all(
                isinstance(message, tuple)
                and len(message) == 2
                and isinstance(message[0], str)
                and isinstance(message[1], str)
                for message in template
            )
        ):
            return template
        raise ValueError(
            f"run.prompts.{alias} must name a string or immutable message template exported by "
            f"utils.prompts; got {name!r}."
        )

    @staticmethod
    def _template_record_fields(template: str) -> tuple[str, ...]:
        """Return top-level named ``str.format`` fields in first-use order."""

        fields: list[str] = []
        try:
            parsed = string.Formatter().parse(template)
            for _literal, field_name, _format_spec, _conversion in parsed:
                if field_name is None:
                    continue
                root = field_name.split(".", 1)[0].split("[", 1)[0]
                if not root or root.isdecimal():
                    raise ValueError(
                        "Evaluation prompt templates must use named fields such as {question}, not positional fields."
                    )
                if root not in fields:
                    fields.append(root)
        except ValueError as error:
            raise ValueError(f"Invalid evaluation prompt template: {error}") from error
        return tuple(fields)

    def _render_evaluation_text(
        self,
        template: str,
        record: Mapping[str, object],
        record_index: int,
        *,
        message_index: int | None = None,
    ) -> str:
        location = f"Evaluation record {record_index}"
        if message_index is not None:
            location += f" message {message_index}"
        for field_name in self._template_record_fields(template):
            if field_name not in record:
                raise ValueError(
                    f"{location} is missing prompt field {field_name!r} required by the template."
                )
            if record[field_name] is None:
                raise ValueError(
                    f"{location} has null prompt field {field_name!r} required by the template."
                )
        try:
            return template.format(**dict(record))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Could not render the evaluation prompt for {location.lower()}: {error}"
            ) from error

    def _render_evaluation_messages(
        self,
        template: str | tuple[tuple[str, str], ...],
        record: Mapping[str, object],
        record_index: int,
    ) -> list[dict[str, str]]:
        message_templates = (("user", template),) if isinstance(template, str) else template
        messages: list[dict[str, str]] = []
        for message_index, (role, content_template) in enumerate(message_templates):
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Evaluation message template {message_index} has unsupported role {role!r}."
                )
            messages.append(
                {
                    "role": role,
                    "content": self._render_evaluation_text(
                        content_template,
                        record,
                        record_index,
                        message_index=message_index,
                    ),
                }
            )
        if not messages:
            raise ValueError("An evaluation message template must contain at least one message.")
        if messages[-1]["role"] != "user":
            raise ValueError(
                "An evaluation message template must end with a user message before generation."
            )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
        if not isinstance(messages, list) or not messages:
            raise TypeError("Rendered evaluation messages must be a non-empty list.")
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"Rendered evaluation message {message_index} must be a mapping.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Rendered evaluation message {message_index} has unsupported role {role!r}."
                )
            if not isinstance(content, str):
                raise TypeError(
                    f"Rendered evaluation message {message_index} content must be a string."
                )

        assert self.tokenizer is not None
        return apply_generation_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _validate_controller_signals(signals: Signals) -> Signals:
        questions, answers = controller_questions_and_answers(signals)
        if "records" in signals:
            records = signals["records"]
            if not isinstance(records, list) or len(records) != len(questions):
                raise ValueError("Controller records must be a list aligned with questions and answers.")
            if any(
                not isinstance(record, Mapping)
                or record.get("question") != questions[index]
                or record.get("answer") != answers[index]
                for index, record in enumerate(records)
            ):
                raise ValueError("Controller records must preserve the matching question and answer fields.")
        return dict(signals)

    def _normalize_return_signals(
        self, require_return: Mapping[str, object] | None
    ) -> dict[str, object]:
        required = copy.deepcopy(DEFAULT_RETURN_SIGNALS)
        if require_return is None:
            return required
        if not isinstance(require_return, Mapping):
            raise TypeError("require_return must be a mapping when provided.")
        for key, value in require_return.items():
            if key == "topk":
                if not isinstance(value, Mapping):
                    raise TypeError("require_return['topk'] must be a mapping.")
                topk = dict(required["topk"])
                topk.update(copy.deepcopy(dict(value)))
                required["topk"] = topk
            else:
                required[str(key)] = copy.deepcopy(value)
        return required

    def _empty_trajectory(self, require_return: Mapping[str, object] | None) -> dict[str, object]:
        required = self._normalize_return_signals(require_return)
        result: dict[str, object] = {}
        for key in ("responses", "response_ids", "prompt_ids"):
            if bool(required.get(key, False)):
                result[key] = []
        torch = _torch()
        if bool(required.get("prompt_indices", False)):
            result["prompt_indices"] = torch.empty((0,), dtype=torch.int32)
        if bool(required.get("response_truncated", False)):
            result["response_truncated"] = torch.empty((0,), dtype=torch.bool)
        if bool(required.get("response_logprobs", False)):
            result["response_logprobs"] = []
        topk = required.get("topk", {})
        if isinstance(topk, Mapping) and bool(topk.get("enabled", False)):
            result["topk_token_ids"] = []
            result["topk_logprobs"] = []
        return result

    def _generation_loop(self) -> None:
        while not self._generation_stop.is_set():
            first = self._request_queue.get()
            if first is None:
                return
            requests = [first]
            while len(requests) < self.max_batches_per_turn:
                try:
                    candidate = self._request_queue.get(timeout=self.collect_timeout_sec)
                except queue.Empty:
                    break
                if candidate is None:
                    self._request_queue.put(None)
                    break
                requests.append(candidate)
            try:
                self._run_batched_generate(requests)
            except Exception as error:
                trace = traceback.format_exc()
                print(f"[Actor GPU {self.gpu_id}] Batched generation failed:\n{trace}", flush=True)
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(error)

    def _run_batched_generate(self, requests: list[_GenerateRequest]) -> None:
        if self.llm is None:
            raise RuntimeError("Actor engine is not initialized.")
        merged_prompts: list[str] = []
        merged_params: list[object] = []
        prompt_counts: list[int] = []
        for request in requests:
            params = self._sampling_params_for_request(
                request.sampling, request.require_return, len(request.prompts)
            )
            request.sampling_params = params
            prompt_counts.append(len(request.prompts))
            merged_prompts.extend(request.prompts)
            merged_params.extend(params)

        self._refresh_lora_request()
        started = time.perf_counter()
        generate = getattr(self.llm, "generate", None)
        if not callable(generate):
            raise TypeError("Actor vLLM engine must provide generate().")
        active_lora_request = self.current_lora_request
        # Prompts already contain the chat template's control tokens, including BOS.
        try:
            outputs = generate(
                merged_prompts,
                merged_params,
                lora_request=active_lora_request,
                use_tqdm=False,
                tokenization_kwargs={"add_special_tokens": False},
            )
        except Exception as error:
            if active_lora_request is None:
                raise
            failed_path = self.current_lora_path
            print(
                f"[Actor GPU {self.gpu_id}] LoRA generation failed; retrying this batch with the base "
                f"model: {failed_path or '<unknown adapter>'} ({error})",
                flush=True,
            )
            self._mark_lora_path_failed(failed_path)
            # A LoRA-enabled vLLM engine accepts a null request as the base
            # model.  Retry exactly once; a base-model failure must remain
            # visible to the caller instead of being silently swallowed.
            outputs = generate(
                merged_prompts,
                merged_params,
                lora_request=None,
                use_tqdm=False,
                tokenization_kwargs={"add_special_tokens": False},
            )
        elapsed = time.perf_counter() - started
        if not isinstance(outputs, list) or len(outputs) != len(merged_prompts):
            raise RuntimeError("vLLM returned a different number of outputs than prompts.")

        total_rollouts = sum(len(getattr(output, "outputs", [])) for output in outputs)
        offset = 0
        for request, prompt_count in zip(requests, prompt_counts):
            request_outputs = outputs[offset : offset + prompt_count]
            request_params = merged_params[offset : offset + prompt_count]
            trajectory, response_tokens, rollout_count, prompt_tokens = self._parse_outputs(
                request_outputs, request_params, request.require_return
            )
            proportional_duration = elapsed * (rollout_count / total_rollouts) if total_rollouts else 0.0
            metadata = self._rollout_metadata(
                proportional_duration, response_tokens, rollout_count, prompt_tokens, len(request_outputs)
            )
            if not request.future.done():
                request.future.set_result((trajectory, metadata))
            offset += prompt_count

        print(
            f"[Actor GPU {self.gpu_id}] Generated prompts={len(merged_prompts)}, "
            f"rollouts={total_rollouts}, lora_step={self.current_lora_step}, time={elapsed:.2f}s",
            flush=True,
        )

    def _sampling_params_for_request(
        self,
        sampling: object,
        require_return: Mapping[str, object],
        prompt_count: int,
    ) -> list[object]:
        if isinstance(sampling, list):
            if len(sampling) != prompt_count:
                raise ValueError(
                    "sampling list length must equal prompts length, "
                    f"got {len(sampling)} != {prompt_count}."
                )
            return [self._one_sampling_params(item, require_return) for item in sampling]
        one = self._one_sampling_params(sampling, require_return)
        return [copy.deepcopy(one) for _ in range(prompt_count)]

    def _one_sampling_params(
        self, sampling: object, require_return: Mapping[str, object]
    ) -> object:
        logprobs = self._required_vllm_logprobs(require_return)
        if isinstance(sampling, Mapping):
            params = copy.deepcopy(dict(sampling))
            if "max_response_length" in params and "max_tokens" not in params:
                params["max_tokens"] = params.pop("max_response_length")
            if "max_new_tokens" in params and "max_tokens" not in params:
                params["max_tokens"] = params.pop("max_new_tokens")
            params["logprobs"] = logprobs
            factory = self._sampling_params_factory
            if factory is None:
                try:
                    from vllm import SamplingParams
                except ImportError as error:  # pragma: no cover - runtime setup issue.
                    raise RuntimeError("vLLM is required to create sampling parameters.") from error
                factory = SamplingParams
                self._sampling_params_factory = factory
            return factory(**params)

        # SamplingParams objects (including CPU-test doubles) are copied so a
        # plugin never observes Actor adding the requested logprob setting.
        copied = copy.deepcopy(sampling)
        try:
            setattr(copied, "logprobs", logprobs)
        except Exception as error:
            raise TypeError("sampling must be a mapping, SamplingParams, or a list of either.") from error
        return copied

    def _required_vllm_logprobs(
        self, require_return: Mapping[str, object] | None
    ) -> int | None:
        if require_return is None:
            require_return = DEFAULT_RETURN_SIGNALS
        if not isinstance(require_return, Mapping):
            raise TypeError("require_return must be a mapping when provided.")
        topk = require_return.get("topk", {})
        topk_enabled = isinstance(topk, Mapping) and bool(topk.get("enabled", False))
        if topk_enabled:
            top_k = int(topk.get("top_k", 0))
            if top_k <= 0:
                raise ValueError("require_return['topk']['top_k'] must be positive when topk is enabled.")
            if top_k > self.max_logprobs:
                raise ValueError(
                    f"Requested top_k={top_k}, but actor.engine.logprobs.max={self.max_logprobs}."
                )
            return top_k
        if bool(require_return.get("response_logprobs", False)):
            if self.default_logprobs is None:
                return None
            return int(self.default_logprobs)
        return None

    def _parse_outputs(
        self,
        outputs: list[object],
        sampling_params: list[object],
        require_return: Mapping[str, object],
    ) -> tuple[dict[str, object], int, int, int]:
        responses: list[str] = []
        response_ids: list[object] = []
        prompt_ids: list[object] = []
        prompt_indices: list[int] = []
        response_truncated: list[bool] = []
        response_logprobs: list[object] = []
        topk_token_ids: list[object] = []
        topk_logprobs: list[object] = []
        response_token_count = 0
        rollout_count = 0
        prompt_token_count = 0
        torch = _torch()

        topk = require_return.get("topk", {})
        topk_enabled = isinstance(topk, Mapping) and bool(topk.get("enabled", False))
        top_k = int(topk.get("top_k", 0)) if topk_enabled else 0

        for prompt_index, (output, sampling) in enumerate(zip(outputs, sampling_params)):
            raw_prompt_ids = getattr(output, "prompt_token_ids", None)
            prompt_token_ids = list(raw_prompt_ids) if raw_prompt_ids else []
            prompt_token_count += len(prompt_token_ids)
            max_tokens = int(getattr(sampling, "max_tokens", 0) or 0)
            output_responses = getattr(output, "outputs", None)
            if output_responses is None:
                raise TypeError("Each vLLM output must provide an outputs sequence.")
            for response in output_responses:
                raw_ids = getattr(response, "token_ids", None)
                if raw_ids is None:
                    raise TypeError("Each vLLM response must provide token_ids.")
                token_ids = [int(token_id) for token_id in raw_ids]
                fallback_text = str(getattr(response, "text", ""))
                responses.append(_decode_generated_response(self.tokenizer, token_ids, fallback_text))
                response_ids.append(torch.tensor(token_ids, dtype=torch.int32))
                prompt_ids.append(torch.tensor(prompt_token_ids, dtype=torch.int32))
                prompt_indices.append(prompt_index)
                response_truncated.append(_response_was_truncated(response, len(token_ids), max_tokens))
                response_token_count += len(token_ids)
                rollout_count += 1

                raw_logprobs = getattr(response, "logprobs", None)
                if bool(require_return.get("response_logprobs", False)):
                    response_logprobs.append(_response_logprobs(raw_logprobs, token_ids))
                if topk_enabled:
                    token_rows: list[list[int]] = []
                    logprob_rows: list[list[float]] = []
                    for step_index in range(len(token_ids)):
                        step = (
                            raw_logprobs[step_index]
                            if raw_logprobs is not None and step_index < len(raw_logprobs)
                            else None
                        )
                        ids, values = _topk_items(step, top_k)
                        token_rows.append(ids)
                        logprob_rows.append(values)
                    if token_rows:
                        topk_token_ids.append(torch.tensor(token_rows, dtype=torch.int32))
                        topk_logprobs.append(torch.tensor(logprob_rows, dtype=torch.float32))
                    else:
                        topk_token_ids.append(torch.empty((0, top_k), dtype=torch.int32))
                        topk_logprobs.append(torch.empty((0, top_k), dtype=torch.float32))

        result: dict[str, object] = {}
        if bool(require_return.get("responses", False)):
            result["responses"] = responses
        if bool(require_return.get("response_ids", False)):
            result["response_ids"] = response_ids
        if bool(require_return.get("prompt_ids", False)):
            result["prompt_ids"] = prompt_ids
        if bool(require_return.get("prompt_indices", False)):
            result["prompt_indices"] = torch.tensor(prompt_indices, dtype=torch.int32)
        if bool(require_return.get("response_truncated", False)):
            result["response_truncated"] = torch.tensor(response_truncated, dtype=torch.bool)
        if bool(require_return.get("response_logprobs", False)):
            result["response_logprobs"] = response_logprobs
        if topk_enabled:
            result["topk_token_ids"] = topk_token_ids
            result["topk_logprobs"] = topk_logprobs
        return result, response_token_count, rollout_count, prompt_token_count

    def _clear_lora_request(self) -> None:
        self.current_lora_step = -1
        self.current_lora_path = ""
        self.current_lora_request = None

    def _mark_lora_path_failed(self, path: str) -> None:
        """Quarantine one bad adapter path and continue with the base model."""

        if path:
            self._failed_lora_paths.add(path)
        self._clear_lora_request()

    @staticmethod
    def _build_lora_request(lora_name: str, lora_int_id: int, lora_local_path: str) -> object:
        """Construct a vLLM request lazily so normal CPU imports stay light."""

        from vllm.lora.request import LoRARequest

        # vLLM renamed this constructor field from ``lora_local_path`` to
        # ``lora_path``.  The current runtime uses the latter, while keeping
        # the narrowly scoped fallback lets an existing environment retain
        # its compatible Actor/adapter handoff behavior.
        try:
            return LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_local_path,
            )
        except TypeError as error:
            if "lora_path" not in str(error):
                raise
            return LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_local_path=lora_local_path,
            )

    def _refresh_lora_request(self) -> None:
        if not self.lora_enabled:
            self._clear_lora_request()
            return

        if self._has_explicit_lora_path:
            step, path = 0, self._explicit_lora_path
            lora_name, lora_int_id = "evaluation_adapter", 1
        else:
            step, path = _latest_lora_adapter(self.config)
            lora_name, lora_int_id = f"step_{step}", step + 1

        if step < 0 or not path:
            self._clear_lora_request()
            return
        if path in self._failed_lora_paths:
            self._clear_lora_request()
            return
        if step == self.current_lora_step and path == self.current_lora_path:
            return
        if not Path(path).is_dir():
            print(
                f"[Actor GPU {self.gpu_id}] LoRA adapter is not a directory; using the base model: {path}",
                flush=True,
            )
            self._mark_lora_path_failed(path)
            return

        try:
            request = self._build_lora_request(lora_name, lora_int_id, path)
        except Exception as error:
            print(
                f"[Actor GPU {self.gpu_id}] Could not load LoRA adapter; using the base model: "
                f"{path} ({error})",
                flush=True,
            )
            self._mark_lora_path_failed(path)
            return

        self.current_lora_request = request
        self.current_lora_step = step
        self.current_lora_path = path
        print(f"[Actor GPU {self.gpu_id}] Loaded LoRA adapter step={step}: {path}", flush=True)

    def _rollout_metadata(
        self, duration_sec: float, response_tokens: int, rollout_count: int, prompt_tokens: int, prompt_count: int
    ) -> dict[str, object]:
        throughput = response_tokens / duration_sec if duration_sec > 0 else 0.0
        mean_tokens = response_tokens / rollout_count if rollout_count else 0.0
        with self._metrics_lock:
            self._completed_rollout_batches += 1
            self._cumulative_duration_sec += float(duration_sec)
            self._cumulative_response_tokens += int(response_tokens)
            self._cumulative_rollouts += int(rollout_count)
            self._cumulative_throughput += float(throughput)
            completed = self._completed_rollout_batches
            average_duration = self._cumulative_duration_sec / completed
            average_tokens = (
                self._cumulative_response_tokens / self._cumulative_rollouts
                if self._cumulative_rollouts
                else 0.0
            )
            average_throughput = self._cumulative_throughput / completed
        return {
            "gpu_id": self.gpu_id,
            "lora_step": self.current_lora_step,
            "completed_rollout_batches": completed,
            "batch_rollout_count": int(rollout_count),
            "cumulative_rollout_count": int(self._cumulative_rollouts),
            "batch_rollout_duration_seconds": float(duration_sec),
            "batch_prompt_tokens_total": int(prompt_tokens),
            "batch_prompt_tokens_mean": float(prompt_tokens / prompt_count if prompt_count else 0.0),
            "batch_response_tokens_total": int(response_tokens),
            "batch_response_tokens_mean": float(mean_tokens),
            "batch_throughput_tokens_per_second": float(throughput),
            "avg_batch_rollout_duration_seconds": float(average_duration),
            "avg_response_tokens_per_response": float(average_tokens),
            "avg_throughput_tokens_per_second": float(average_throughput),
        }

    def _redis_client(self) -> object:
        try:
            import redis
        except ImportError as error:  # pragma: no cover - production setup issue.
            raise RuntimeError("redis-py is required to run the Actor worker.") from error
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        url = redis_cfg.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("run.redis.url must be a non-empty string.")
        return redis.Redis.from_url(url, decode_responses=False)

    def run_forever(self, stop_event: object) -> None:
        """Read train/eval inboxes, batch requests, and route detached results."""

        if not callable(getattr(stop_event, "is_set", None)):
            raise TypeError("stop_event must provide is_set().")
        client = self._redis_client()
        keys = self.queue_keys()
        poll_timeout = float(self.actor_cfg.get("queue_poll_timeout_sec", 1.0))
        if poll_timeout <= 0:
            poll_timeout = 1.0
        input_kind = {keys["actor_train"]: "train", keys["actor_eval"]: "eval"}
        print(
            f"[Actor GPU {self.gpu_id}] Listening on {keys['actor_train']} and {keys['actor_eval']}; "
            f"max_batches_per_turn={self.max_batches_per_turn}.",
            flush=True,
        )

        while not stop_event.is_set():
            try:
                popped = client.brpop(list(input_kind), timeout=poll_timeout)
            except Exception as error:
                print(f"[Actor GPU {self.gpu_id}] Redis receive failed: {error}; retrying.", flush=True)
                self._wait_or_stop(stop_event, 1.0)
                continue
            if not popped:
                continue
            raw_key, raw_payload = popped
            queue_name = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            task_type = input_kind.get(queue_name)
            if task_type is None:
                self._publish_event(client, f"received an unknown Actor queue key: {queue_name}")
                continue
            batches = [(task_type, raw_payload)]
            # Keep a redis pull group homogeneous.  Train and evaluation have
            # independent queues and never share a downstream result key.
            for _ in range(self.max_batches_per_turn - 1):
                try:
                    extra = client.rpop(queue_name)
                except Exception:
                    break
                if extra is None:
                    break
                batches.append((task_type, extra))
            self._process_redis_batches(client, batches, stop_event=stop_event)

    def _process_redis_batches(
        self,
        client: object,
        batches: list[tuple[str, object]],
        *,
        stop_event: object | None = None,
    ) -> None:
        """Process one bounded Actor turn while holding this GPU's Gate.

        Redis decoding happens before acquiring the permit.  The critical
        section starts only when the merged vLLM generation work begins and
        ends after its detached outputs have been published downstream.
        ``vLLM.generate()`` is synchronous with its EngineCore, so it is the
        completion boundary here; a ``torch.cuda.synchronize()`` in this
        client process would not synchronize vLLM's separate EngineCore.
        """

        decoded: list[tuple[str, Signals]] = []
        for task_type, payload in batches:
            try:
                signals = pickle.loads(payload)
                if not isinstance(signals, Mapping):
                    raise TypeError("Redis payload did not contain a signal mapping.")
                decoded.append((task_type, dict(signals)))
            except Exception as error:
                self._publish_event(client, f"could not decode Actor input: {error}")

        if not decoded:
            return
        waiting_started = time.perf_counter()
        if not self.gpu_gate.acquire(stop_event):
            return
        gate_wait_seconds = time.perf_counter() - waiting_started
        turn_started = time.perf_counter()
        completed_count = 0
        try:
            with ThreadPoolExecutor(max_workers=len(decoded)) as executor:
                futures = {
                    executor.submit(
                        self.process_training_batch if task_type == "train" else self.process_evaluation_batch,
                        signals,
                    ): (task_type, time.perf_counter())
                    for task_type, signals in decoded
                }
                completed: list[tuple[str, Signals]] = []
                for future in as_completed(futures):
                    task_type, batch_started = futures[future]
                    try:
                        processed = future.result()
                        timed = self._with_worker_timing(
                            processed,
                            worker="actor",
                            processing_seconds=time.perf_counter() - batch_started,
                            gate_wait_seconds=gate_wait_seconds,
                        )
                        completed.append(
                            (
                                task_type,
                                timed,
                            )
                        )
                    except Exception as error:
                        self._publish_event(client, f"{task_type} batch failed: {error}")

            if not completed:
                return
            pipeline = client.pipeline()
            for task_type, signals in completed:
                route = self.route_output(task_type)
                safe_signals = _deep_cpu_detach(signals)
                pipeline.lpush(route, pickle.dumps(safe_signals, protocol=pickle.HIGHEST_PROTOCOL))
            pipeline.execute()
            completed_count = len(completed)
        finally:
            gate_hold_seconds = time.perf_counter() - turn_started
            self.gpu_gate.release()
            print(
                f"[Actor GPU {self.gpu_id}] turn batches={completed_count}/{len(decoded)}, "
                f"gate_wait={gate_wait_seconds:.3f}s, gate_hold={gate_hold_seconds:.3f}s.",
                flush=True,
            )
            self.gpu_gate.yield_after_release(stop_event, delay_sec=2.0)

    @staticmethod
    def _with_worker_timing(
        signals: Signals,
        *,
        worker: str,
        processing_seconds: float,
        gate_wait_seconds: float,
    ) -> Signals:
        """Attach framework timing without changing algorithm-owned signals."""

        result: Signals = dict(signals)
        raw_timing = result.get("timing", {})
        timing = dict(raw_timing) if isinstance(raw_timing, Mapping) else {}
        raw_workers = timing.get("workers", {})
        workers = dict(raw_workers) if isinstance(raw_workers, Mapping) else {}
        workers[worker] = {
            "processing_seconds": max(0.0, float(processing_seconds)),
            "gate_wait_seconds": max(0.0, float(gate_wait_seconds)),
        }
        timing["workers"] = workers
        result["timing"] = timing
        return result

    def _publish_event(self, client: object, message: str) -> None:
        event = {
            "source": "actor",
            "gpu_id": self.gpu_id,
            "message": message,
            "timestamp": time.time(),
        }
        try:
            client.lpush(
                self.queue_keys()["controller_events"],
                pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL),
            )
        except Exception as error:
            print(f"[Actor GPU {self.gpu_id}] Could not publish event: {error}", flush=True)

    @staticmethod
    def _wait_or_stop(stop_event: object, seconds: float) -> None:
        wait = getattr(stop_event, "wait", None)
        if callable(wait):
            wait(seconds)
        else:  # pragma: no cover - multiprocessing Event always provides wait.
            time.sleep(seconds)

    def _shutdown_vllm_engine_core(self, llm: object) -> None:
        """Best-effort explicit shutdown for a fully idle vLLM EngineCore.

        vLLM 0.19's ``UniProcExecutor.shutdown`` does not call
        ``destroy_model_parallel`` or ``destroy_distributed_environment``.
        Calling the public utility RPC first fixes that omission.  The bytes
        form avoids requiring ``VLLM_ALLOW_INSECURE_SERIALIZATION`` and is
        supported by vLLM's ``serial_utils.run_method`` implementation.
        ``EngineCoreClient.shutdown`` still owns process/KV-cache teardown and
        must run even when the utility RPC cannot be completed.
        """

        collective_rpc = getattr(llm, "collective_rpc", None)
        if callable(collective_rpc):
            try:
                import cloudpickle

                payload = cloudpickle.dumps(_destroy_vllm_worker_process_groups)
                collective_rpc(payload)
                print(
                    f"[Actor GPU {self.gpu_id}] Destroyed vLLM EngineCore distributed state.",
                    flush=True,
                )
            except Exception as error:
                # A partially initialized engine or an incompatible future
                # vLLM version must not prevent the ordinary client shutdown.
                # Make the fallback visible rather than silently hiding a
                # potentially meaningful distributed-cleanup regression.
                print(
                    f"[Actor GPU {self.gpu_id}] WARNING: could not destroy vLLM EngineCore "
                    f"distributed state explicitly: {error}",
                    flush=True,
                )
        else:
            print(
                f"[Actor GPU {self.gpu_id}] WARNING: vLLM instance exposes no collective_rpc(); "
                "using its ordinary EngineCore shutdown only.",
                flush=True,
            )

        llm_engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if not callable(shutdown):
            print(
                f"[Actor GPU {self.gpu_id}] WARNING: vLLM EngineCore client exposes no shutdown(); "
                "relying on object finalization.",
                flush=True,
            )
            return
        try:
            shutdown()
            print(f"[Actor GPU {self.gpu_id}] Shut down vLLM EngineCore client.", flush=True)
        except Exception as error:
            print(
                f"[Actor GPU {self.gpu_id}] WARNING: vLLM EngineCore client shutdown failed: {error}",
                flush=True,
            )

    def _release_engine(self) -> None:
        generation_stopped = self.stop_generation_loop()
        self.current_lora_request = None
        self.current_lora_path = ""
        self.current_lora_step = -1
        self.tokenizer = None
        llm = self.llm
        if llm is not None:
            if generation_stopped:
                self._shutdown_vllm_engine_core(llm)
            else:
                # Keep the old finalizer fallback for an abnormally blocked
                # request thread.  Explicit utility RPC is unsafe until that
                # thread has left ``LLM.generate``.
                print(
                    f"[Actor GPU {self.gpu_id}] WARNING: leaving vLLM EngineCore cleanup to finalization "
                    "because generation did not quiesce.",
                    flush=True,
                )
        self.llm = None
        gc.collect()
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        """Release generation resources during a worker retry or shutdown."""

        self._release_engine()


PluginFactory = Callable[[object], AlgorithmPlugin]


def _build_plugin(factory: PluginFactory | None, tokenizer: object) -> AlgorithmPlugin | None:
    if factory is None:
        return None
    plugin = factory(tokenizer)
    if not isinstance(plugin, AlgorithmPlugin):
        raise TypeError("plugin_factory must return an AlgorithmPlugin.")
    return plugin


class ActorManager:
    """Launch and supervise one independent Actor process per configured GPU."""

    def __init__(
        self,
        config: Mapping[str, object],
        plugin_factory: PluginFactory | None = None,
        *,
        gpu_gates: GpuGateRegistry | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        self.config = copy.deepcopy(dict(config))
        self.plugin_factory = plugin_factory
        self.gpu_gates = gpu_gates
        actor_cfg = _mapping(self.config.get("actor"), "actor")
        run_cfg = _mapping(self.config.get("run"), "run")
        runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
        engine_cfg = _mapping(actor_cfg.get("engine"), "actor.engine")
        raw_gpu_ids = actor_cfg.get("gpu_ids", [])
        if not isinstance(raw_gpu_ids, list):
            raise ValueError("actor.gpu_ids must be a list of GPU ids.")
        self.gpu_ids = [self._validated_gpu_id(gpu_id) for gpu_id in raw_gpu_ids]
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("actor.gpu_ids may not contain duplicates.")
        if not self.gpu_ids:
            raise ValueError("actor.gpu_ids must contain at least one GPU id.")
        if int(engine_cfg.get("tensor_parallel_size", 1)) != 1:
            raise ValueError(
                "actor.engine.tensor_parallel_size must be 1; use multiple actor.gpu_ids for multi-GPU Actors."
            )
        minimum = float(runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        if minimum < 0:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb must be non-negative.")
        start_method = str(runtime_cfg.get("process_start_method", "spawn"))
        self._context = mp.get_context(start_method)
        self.stop_event = self._context.Event()
        self.processes: dict[int, object] = {}
        # A Manager supervisor records a terminal Worker exit here instead of
        # replacing that Worker.  In particular, a process killed while it
        # owns a shared-GPU semaphore may leave that permit unavailable; a
        # replacement Actor would then wait forever.  Controller observes
        # this field and fails the whole run through its normal cleanup path.
        self.fatal_error: str | None = None
        # A Worker sets its own event only after vLLM, its tokenizer, and the
        # selected algorithm plugin are usable.  Controller uses this to
        # serialize initial model residency on a shared local GPU.
        self.ready_events: dict[int, object] = {}

    @staticmethod
    def _validated_gpu_id(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("GPU ids must be non-negative integers.")
        try:
            gpu_id = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("GPU ids must be non-negative integers.") from error
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative integers.")
        return gpu_id

    def start(self) -> None:
        for gpu_id in self.gpu_ids:
            self.processes[gpu_id] = self._start_worker(gpu_id)
        print(f"[Actor Manager] Started {len(self.processes)} Actor worker(s).", flush=True)

    def _start_worker(self, gpu_id: int) -> object:
        ready_event = self._context.Event()
        self.ready_events[gpu_id] = ready_event
        process = self._context.Process(
            target=_actor_worker_main,
            args=(
                gpu_id,
                self.config,
                self.plugin_factory,
                self.stop_event,
                ready_event,
                self.gpu_gates.for_gpu(gpu_id) if self.gpu_gates is not None else None,
            ),
            name=f"FrugalOPL-Actor-GPU{gpu_id}",
        )
        process.start()
        return process

    def _uses_shared_gpu_gate(self, gpu_id: int) -> bool:
        """Whether this Actor's physical GPU has a cross-Worker permit."""

        return self.gpu_gates is not None and self.gpu_gates.for_gpu(gpu_id).is_shared

    def _record_terminal_failure(self, message: str) -> None:
        """Make a Worker exit visible to Controller and stop sibling Actors."""

        if self.fatal_error is None:
            self.fatal_error = message
            print(f"[Actor Manager FATAL] {message}", flush=True)
        self.stop_event.set()

    def wait_until_ready(self, *, timeout_sec: float = 300.0) -> None:
        """Block until every initially launched Actor can accept requests.

        This does not serialize independent Actors on different GPUs.  It is
        consumed by Controller between *worker types*, so a one-GPU run does
        not transiently load vLLM alongside Teacher or Learner initialization.
        """

        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + float(timeout_sec)
        pending = set(self.processes)
        while pending:
            for gpu_id in tuple(pending):
                event = self.ready_events.get(gpu_id)
                is_set = getattr(event, "is_set", None)
                if callable(is_set) and is_set():
                    pending.remove(gpu_id)
                    continue
                process = self.processes.get(gpu_id)
                alive = getattr(process, "is_alive", None)
                if not callable(alive) or not alive():
                    exitcode = getattr(process, "exitcode", None)
                    raise RuntimeError(
                        f"Actor GPU {gpu_id} exited before becoming ready (exitcode={exitcode})."
                    )
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waiting = ", ".join(str(gpu_id) for gpu_id in sorted(pending))
                raise TimeoutError(f"Timed out waiting for Actor GPU(s) {waiting} to become ready.")
            self.stop_event.wait(min(0.1, remaining))

    def join(self) -> None:
        run_cfg = _mapping(self.config.get("run"), "run")
        runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
        restart_cfg = _mapping(runtime_cfg.get("restart", {}), "run.runtime.restart")
        restart_enabled = bool(restart_cfg.get("enabled", True))
        delay = max(0.0, float(restart_cfg.get("worker_restart_delay_sec", 5)))
        poll_interval = max(0.05, float(restart_cfg.get("manager_poll_interval_sec", 2)))
        try:
            while not self.stop_event.is_set() and self.processes:
                for gpu_id, process in list(self.processes.items()):
                    if process.is_alive():
                        continue
                    exitcode = process.exitcode
                    process.join(timeout=0.1)
                    if self._uses_shared_gpu_gate(gpu_id):
                        self._record_terminal_failure(
                            f"Actor GPU {gpu_id} exited unexpectedly (exitcode={exitcode}) while using a shared "
                            "GPU gate; automatic restart is unsafe because a hard exit can leave its semaphore "
                            "locked."
                        )
                        return
                    if not restart_enabled:
                        self._record_terminal_failure(
                            f"Actor GPU {gpu_id} exited unexpectedly (exitcode={exitcode}) and worker restart is "
                            "disabled."
                        )
                        return
                    if self.stop_event.is_set():
                        return
                    print(
                        f"[Actor Manager] Actor GPU {gpu_id} exited (exitcode={exitcode}); restarting.",
                        flush=True,
                    )
                    if self.stop_event.wait(delay):
                        break
                    self.processes[gpu_id] = self._start_worker(gpu_id)
                self.stop_event.wait(poll_interval)
        except KeyboardInterrupt:
            self.stop()

    def request_stop(self) -> None:
        """Broadcast cooperative cancellation without blocking on teardown.

        Controller calls this for every Worker role before any Manager starts
        joining children.  That is essential for a shared card: a Learner
        waiting behind Actor's vLLM cleanup must already know that it should
        leave its Redis loop and clean up as soon as the gate is released.
        """

        self.stop_event.set()

    def _shutdown_grace_sec(self) -> float:
        """Return the cooperative grace period for this Manager's children."""

        if any(self._uses_shared_gpu_gate(gpu_id) for gpu_id in self.processes):
            return _SHARED_GPU_SHUTDOWN_GRACE_SEC
        return _EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC

    @staticmethod
    def _join_processes(processes: tuple[object, ...], *, timeout_sec: float) -> None:
        """Join a set of already-started children within one total deadline."""

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)

    @staticmethod
    def _alive_processes(processes: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(process for process in processes if process.is_alive())

    @staticmethod
    def _signal_processes(processes: tuple[object, ...], method_name: str) -> None:
        """Invoke one process-control primitive on each still-owned child."""

        for process in processes:
            method = getattr(process, method_name, None)
            if method_name == "kill" and not callable(method):
                # ``Process.kill`` is available in the supported WSL runtime.
                # Keep a terminate fallback for alternate process adapters
                # rather than silently leaving a child alive.
                method = getattr(process, "terminate", None)
            if callable(method):
                method()

    def finalize_stop(self) -> None:
        """Join owned Actor children, escalating only after a bounded grace.

        The normal path deliberately does *not* call ``terminate`` while an
        Actor might be holding a shared GPU gate.  Its vLLM teardown owns that
        permit and needs time to release it before Learner/Teacher cleanup can
        acquire the same card.  The final kill fallback is only for a child
        stuck in native code; it is followed by joins so this Manager never
        abandons one of its own subprocesses.
        """

        processes = tuple(self.processes.values())
        if not processes:
            return

        grace_sec = self._shutdown_grace_sec()
        self._join_processes(processes, timeout_sec=grace_sec)
        remaining = self._alive_processes(processes)
        if not remaining:
            return

        print(
            f"[Actor Manager] Cooperative shutdown exceeded {grace_sec:.0f}s; "
            f"sending terminate to {len(remaining)} Actor worker(s).",
            flush=True,
        )
        self._signal_processes(remaining, "terminate")
        self._join_processes(remaining, timeout_sec=_TERMINATE_SHUTDOWN_GRACE_SEC)
        remaining = self._alive_processes(remaining)
        if not remaining:
            return

        # Worker signal handlers intentionally translate SIGTERM into a
        # cooperative stop request.  A truly stuck native child may therefore
        # ignore it indefinitely; SIGKILL is the final, manager-owned escape.
        print(
            f"[Actor Manager] Terminate grace exceeded; killing {len(remaining)} Actor worker(s).",
            flush=True,
        )
        self._signal_processes(remaining, "kill")
        self._join_processes(remaining, timeout_sec=_KILL_SHUTDOWN_JOIN_SEC)
        remaining = self._alive_processes(remaining)
        if remaining:
            raise RuntimeError(
                "Actor Manager could not reap all owned workers after forced shutdown; "
                f"{len(remaining)} process(es) remain alive."
            )

    def stop(self) -> None:
        """Standalone Manager shutdown: request first, then finalize owned children."""

        self.request_stop()
        self.finalize_stop()


def _actor_worker_main(
    gpu_id: int,
    config: Mapping[str, object],
    plugin_factory: PluginFactory | None,
    stop_event: object,
    ready_event: object,
    gpu_gate: GpuGate | None,
) -> None:
    # Set this before vLLM, transformers, or torch are imported in the child.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Direct manager use has no Controller-created registry, but it must still
    # follow the same ownership path as a Controller-managed worker.  Keeping
    # one stable object here also makes the initialization permit and each
    # subsequent generation-turn permit refer to the same physical GPU.
    worker_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=gpu_id)

    def request_stop(_signum: int, _frame: object) -> None:
        set_stop = getattr(stop_event, "set", None)
        if callable(set_stop):
            set_stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run_cfg = _mapping(config.get("run"), "run")
    runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
    restart_cfg = _mapping(runtime_cfg.get("restart", {}), "run.runtime.restart")
    retry_delay = max(0.0, float(restart_cfg.get("worker_restart_delay_sec", 5)))

    while not stop_event.is_set():
        actor: Actor | None = None
        try:
            # Engine construction loads vLLM weights, allocates its KV cache,
            # and can create a CUDA context.  It therefore needs the same
            # physical-GPU ownership as an ordinary generation turn.  This is
            # especially important when a worker restarts after its peers are
            # already serving work on the shared card.
            if not worker_gate.acquire(stop_event):
                break
            try:
                print(f"[Actor GPU {gpu_id}] Initializing vLLM engine.", flush=True)
                actor = Actor(config, gpu_id=gpu_id, gpu_gate=worker_gate)
                actor.initialize_engine()
                if actor.tokenizer is None:
                    raise RuntimeError("Actor initialized without a tokenizer.")
                plugin = _build_plugin(plugin_factory, actor.tokenizer)
                if plugin is not None:
                    actor.set_plugin(plugin)
            except BaseException as error:
                # A plugin/tokenizer error can happen after vLLM has already
                # allocated GPU state.  Tear that state down before yielding
                # the permit; otherwise ``empty_cache`` may race a peer
                # Worker that acquires the card immediately after us.
                if actor is not None:
                    actor_to_close, actor = actor, None
                    actor_to_close.close()
                raise
            finally:
                # ``initialize_engine`` can fail after it has allocated part
                # of an engine.  Always return the permit after any partial
                # engine cleanup, while never releasing if shutdown won
                # before ``acquire`` above.
                worker_gate.release()
            actor.start_generation_loop()
            print(f"[Actor GPU {gpu_id}] Engine ready.", flush=True)
            mark_ready = getattr(ready_event, "set", None)
            if callable(mark_ready):
                mark_ready()
            actor.run_forever(stop_event)
        except Exception as error:
            print(
                f"[Actor GPU {gpu_id} FATAL] {error}\n{traceback.format_exc()}",
                flush=True,
            )
            Actor._wait_or_stop(stop_event, retry_delay)
        finally:
            if actor is not None:
                # ``Actor.close`` may release vLLM allocations and empty a
                # CUDA cache.  Shutdown has usually set ``stop_event`` by
                # this point, so deliberately do *not* pass it to acquire:
                # cleanup must wait for an in-flight peer rather than racing
                # it or being skipped because cancellation is already set.
                actor_to_close, actor = actor, None
                if worker_gate.acquire():
                    try:
                        actor_to_close.close()
                    finally:
                        worker_gate.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FrugalOPL's generic vLLM Actor workers.")
    parser.add_argument(
        "--framework-config",
        required=True,
        help="Path to a framework profile YAML.",
    )
    parser.add_argument(
        "--algorithm-config",
        default=None,
        help="Optional algorithm overlay YAML file.",
    )
    args = parser.parse_args()
    config = load_merged_config(args.framework_config, args.algorithm_config)
    factory = plugin_factory_from_config(config)
    manager = ActorManager(config, plugin_factory=factory)
    manager.start()
    manager.join()


if __name__ == "__main__":
    main()


__all__ = [
    "Actor",
    "ActorManager",
    "DEFAULT_RETURN_SIGNALS",
    "EVALUATION_RETURN_SIGNALS",
    "PluginFactory",
    "gpu_memory_headroom_is_safe",
    "load_merged_config",
    "load_yaml_config",
    "load_yaml_config_tree",
    "merge_configs",
]
