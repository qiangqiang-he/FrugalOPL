"""The generic Transformers + DeepSpeed Learner worker.

The Learner owns model residency, LoRA switching, model forwards, answer
probes, micro-batching, distributed placement, backward passes, optimizer
steps, and adapter publication.  Algorithm plugins only see ordinary signal
dictionaries and call two worker-local helpers inherited from
``AlgorithmPlugin``:

``self.answer_probe(batch, probability_difference_threshold=...)``
    Runs over a complete logical rollout batch.  Internally the Learner may
    shard no-grad probe work across ranks and small physical chunks.

``self.compute_logits(batch, mode=...)``
    Scores a complete logical batch without gradients in the signal hook,
    or the current single-rollout micro-batch inside ``compute_objective``.
    It returns token-aligned signals, never full-vocabulary logits.

No algorithm code receives a Learner, Redis client, rank, CUDA device, or
DeepSpeed engine.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import signal
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

from algorithm_plugins import plugin_factory_from_config
from algorithm_plugins.algorithm_base import (
    AlgorithmPlugin,
    Metrics,
    Signals,
    _learner_plugin_scope,
)
from engine.gpu_gate import GpuGate, GpuGateRegistry
from engine.pipeline import resolve_training_pipeline
from utils.answer_verifier import verify_response_answer


ComputeMode = Literal["reference", "policy_no_grad", "policy_grad"]

DEFAULT_LOGIT_SIGNALS: dict[str, object] = {
    "response_logprobs": True,
    "response_entropy": False,
    "entropy_requires_grad": True,
    "topk": {"enabled": False, "top_k": 0},
}

_ROLLOUT_CORE_FIELDS = frozenset(
    {
        "responses",
        "prompt_ids",
        "response_ids",
        "prompt_indices",
        "response_truncated",
        "response_logprobs",
        "topk_token_ids",
        "topk_logprobs",
    }
)

_LOGIT_SIGNAL_KEYS = frozenset(DEFAULT_LOGIT_SIGNALS)
_TOPK_SIGNAL_KEYS = frozenset({"enabled", "top_k", "k"})

# On a shared physical GPU, a normal Learner shutdown may be waiting for
# Actor's vLLM or Teacher's model teardown to release the execution gate
# before it can destroy NCCL and free its own CUDA allocations.  This explicit
# lifecycle grace avoids killing a Worker that is doing precisely that normal
# cleanup.  It is intentionally not a user-facing training hyperparameter.
_EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC = 5.0
_SHARED_GPU_SHUTDOWN_GRACE_SEC = 30.0
_TERMINATE_SHUTDOWN_GRACE_SEC = 5.0
_KILL_SHUTDOWN_JOIN_SEC = 5.0


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


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


def _latest_lora_adapter(config: Mapping[str, object]) -> tuple[int, Path | None]:
    """Return the newest fully published adapter path, if one exists."""

    root = _lora_adapter_root(config)
    if not root.is_dir():
        return -1, None
    best_step = -1
    best_path: Path | None = None
    for child in root.iterdir():
        if not child.is_dir() or child.name.endswith("_tmp"):
            continue
        match = re.fullmatch(r"step_(\d+)", child.name)
        if match is None:
            continue
        adapter = child / "adapter_lora"
        step = int(match.group(1))
        if adapter.is_dir() and step > best_step:
            best_step = step
            best_path = adapter
    return best_step, best_path


def _deep_cpu_detach(value: object) -> object:
    """Detach nested tensors for Redis or result serialization."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _deep_cpu_detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_cpu_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_cpu_detach(item) for item in value)
    return value


def _move_tensors_to_device(value: object, device: torch.device) -> object:
    """Recursively move a collated plugin batch onto the Learner device.

    Actor and no-grad Learner signals deliberately travel as CPU tensors over
    Redis.  A training plugin, however, must be able to combine them directly
    with ``policy_grad`` logits without learning about device placement.  This
    helper is applied only to the short-lived micro-batch passed to
    ``compute_objective``; the logical Redis payload remains CPU-resident.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value


def _as_1d_long(value: object, path: str) -> torch.Tensor:
    """Coerce one ragged token sequence to a CPU ``long`` vector."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().to(dtype=torch.long)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensor = torch.tensor(list(value), dtype=torch.long)
    else:
        raise TypeError(f"{path} must be a one-dimensional token tensor or sequence.")
    if tensor.ndim != 1:
        raise ValueError(f"{path} must be one-dimensional; got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _as_ragged_token_list(value: object, path: str) -> list[torch.Tensor]:
    """Normalize a logical rollout batch into CPU one-dimensional token tensors.

    A rank-two tensor is accepted only as a convenient representation of
    already uniform, complete rollout sequences.  It is never interpreted as
    padded data: every row is retained verbatim and later becomes its own
    physical ``[1, L]`` forward.
    """

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 1:
            return [tensor.to(dtype=torch.long).contiguous()]
        if tensor.ndim == 2:
            return [row.to(dtype=torch.long).contiguous() for row in tensor]
        raise ValueError(f"{path} tensor must be rank 1 or 2; got rank {tensor.ndim}.")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_as_1d_long(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} must be a tensor or sequence of token sequences.")


def _tensor_to_python_bool_list(value: object, *, length: int, path: str) -> list[bool]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"{path} must be a boolean tensor or sequence.")
    if len(values) != length:
        raise ValueError(f"{path} must have length {length}; got {len(values)}.")
    return [bool(item) for item in values]


def _tensor_to_python_int_list(value: object, *, length: int, path: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"{path} must be an integer tensor or sequence.")
    if len(values) != length:
        raise ValueError(f"{path} must have length {length}; got {len(values)}.")
    return [int(item) for item in values]


def _normalize_logit_signals(value: Mapping[str, object] | None) -> dict[str, object]:
    """Merge a partial token-signal request with the Learner defaults."""

    required = copy.deepcopy(DEFAULT_LOGIT_SIGNALS)
    if value is None:
        return required
    if not isinstance(value, Mapping):
        raise TypeError("return_signals must be a mapping when provided.")
    unknown = set(value).difference(_LOGIT_SIGNAL_KEYS)
    if unknown:
        raise ValueError(
            "compute_logits return_signals supports only "
            f"{sorted(_LOGIT_SIGNAL_KEYS)}; got unsupported fields {sorted(map(str, unknown))}."
        )
    for key, item in value.items():
        if key == "topk":
            if not isinstance(item, Mapping):
                raise TypeError("return_signals['topk'] must be a mapping.")
            unknown_topk = set(item).difference(_TOPK_SIGNAL_KEYS)
            if unknown_topk:
                raise ValueError(
                    "return_signals.topk supports only "
                    f"{sorted(_TOPK_SIGNAL_KEYS)}; got unsupported fields "
                    f"{sorted(map(str, unknown_topk))}."
                )
            nested = dict(required["topk"])
            nested.update(copy.deepcopy(dict(item)))
            required["topk"] = nested
        else:
            required[str(key)] = copy.deepcopy(item)

    topk = _mapping(required.get("topk", {}), "return_signals.topk")
    enabled = bool(topk.get("enabled", False))
    top_k = int(topk.get("top_k", topk.get("k", 0)) or 0)
    if enabled and top_k <= 0:
        raise ValueError("return_signals.topk.top_k must be positive when topk is enabled.")
    required["topk"] = {"enabled": enabled, "top_k": top_k}
    return required


def _torch_dtype(value: object) -> torch.dtype:
    name = str(value or "bfloat16").strip().lower()
    aliases: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported run.policy.dtype: {value!r}.")
    return aliases[name]


def _metric_number(value: object) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _balanced_indices(total: int, world_size: int, rank: int) -> list[int]:
    if total < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed index partition.")
    return list(range(rank, total, world_size))


@dataclass(frozen=True)
class _AnswerProbe:
    """One fully tokenized gold-answer suffix and its scored local offsets."""

    token_ids: tuple[int, ...]
    answer_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class _ProbePlan:
    """One truncated rollout prepared for a packed head/tail probe forward."""

    rollout_index: int
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    probe: _AnswerProbe


def _tokenizer_result_ids(value: object, path: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    else:
        values = value
    if isinstance(values, Sequence) and values and isinstance(values[0], Sequence) and not isinstance(
        values[0], (str, bytes)
    ):
        if len(values) != 1:
            raise ValueError(f"{path} must contain exactly one token sequence.")
        values = values[0]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{path} must be a token sequence.")
    return [int(token_id) for token_id in values]


def _decode_ids(tokenizer: object, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("Learner tokenizer must provide decode().")
    return str(
        decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _probe_offsets_fallback(
    tokenizer: object,
    probe_ids: Sequence[int],
    probe_text: str,
) -> list[tuple[int, int]]:
    """Recover token character spans when a tokenizer has no offset mapping."""

    boundaries = [0]
    for end in range(1, len(probe_ids) + 1):
        decoded = _decode_ids(tokenizer, probe_ids[:end])
        boundaries.append(len(decoded) if probe_text.startswith(decoded) else boundaries[-1])
    if boundaries[-1] != len(probe_text):
        raise ValueError("The answer probe cannot be aligned to tokenizer output.")
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _build_answer_probe(tokenizer: object, answer: object) -> _AnswerProbe:
    """Tokenize the fixed answer suffix with minimum-overlap target coverage."""

    answer_text = str(answer)
    if not answer_text:
        raise ValueError("Cannot build an answer probe for an empty gold answer.")
    prefix = "Therefore, the answer is \\boxed{"
    probe_text = f"{prefix}{answer_text}}}"
    answer_start = len(prefix)
    answer_end = answer_start + len(answer_text)

    tokenizer_call = getattr(tokenizer, "__call__", None)
    if not callable(tokenizer_call):
        raise TypeError("Learner tokenizer must be callable to build an answer probe.")
    encoded = tokenizer_call(
        probe_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("Tokenizer answer-probe output must be a mapping.")
    probe_ids = _tokenizer_result_ids(encoded.get("input_ids"), "tokenizer input_ids")
    if not probe_ids:
        raise ValueError("Answer-probe tokenization produced no tokens.")
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        offset_pairs = _probe_offsets_fallback(tokenizer, probe_ids, probe_text)
    else:
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.detach().cpu().tolist()
        if isinstance(offsets, Sequence) and offsets and isinstance(offsets[0], Sequence):
            first = offsets[0]
            if isinstance(first, Sequence) and first and isinstance(first[0], Sequence):
                if len(offsets) != 1:
                    raise ValueError("Tokenizer returned multiple answer-probe offset rows.")
                offsets = first
        if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
            raise TypeError("Tokenizer offset_mapping must be a sequence.")
        offset_pairs = [(int(pair[0]), int(pair[1])) for pair in offsets]
    if len(offset_pairs) != len(probe_ids):
        raise ValueError("Answer-probe token IDs and offsets have different lengths.")
    if _decode_ids(tokenizer, probe_ids) != probe_text:
        raise ValueError("Answer-probe tokens do not round-trip through the tokenizer.")

    # An overlap test is intentional: a BPE token such as ``{42`` contains
    # part of the box marker and part of the answer, so it is the minimum
    # valid token coverage for answer ``42``.
    positions = tuple(
        index
        for index, (start, end) in enumerate(offset_pairs)
        if end > answer_start and start < answer_end
    )
    if not positions or positions[0] <= 0:
        raise ValueError("Answer-probe answer tokens need an in-branch causal predecessor.")
    return _AnswerProbe(token_ids=tuple(probe_ids), answer_token_positions=positions)


class Learner:
    """One rank of the generic policy-training Worker.

    ``LearnerManager`` constructs one instance per configured GPU.  Rank zero
    receives Redis payloads and broadcasts them; all ranks execute the same
    Learner hook over the complete logical batch, then receive independent
    training micro-batches for ``compute_objective``.

    The optional ``device`` and pre-populated ``model``/``tokenizer`` fields
    make CPU-only contract tests possible without loading a real model.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        plugin: AlgorithmPlugin | None = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str | None = None,
        gpu_id: int | None = None,
        gpu_gate: GpuGate | None = None,
        node_id: str = "learner",
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        self.config: dict[str, object] = copy.deepcopy(dict(config))
        self.run_cfg = _mapping(self.config.get("run"), "run")
        self.policy_cfg = _mapping(self.run_cfg.get("policy"), "run.policy")
        self.learner_cfg = _mapping(self.config.get("learner"), "learner")
        self.pipeline = resolve_training_pipeline(self.config)
        self.pipeline_node = self.pipeline.node(node_id)
        self.node_id = self.pipeline_node.node_id
        self.upstream_node_id = self.pipeline.predecessor_of(self.node_id)
        self.plugin_hook = self.pipeline_node.plugin_hook
        self.objective_fn = self.pipeline_node.objective_fn
        self.cpu_offload = self.pipeline_node.cpu_offload
        self.execution_cfg = _mapping(self.learner_cfg.get("execution", {}), "learner.execution")
        self.distributed_cfg = _mapping(self.learner_cfg.get("distributed", {}), "learner.distributed")
        self.checkpoint_cfg = _mapping(self.learner_cfg.get("checkpoint", {}), "learner.checkpoint")
        self.runtime_cfg = _mapping(self.run_cfg.get("runtime", {}), "run.runtime")
        if "micro_batch_size" in self.execution_cfg:
            raise ValueError(
                "learner.execution.micro_batch_size has been removed; "
                "FrugalOPL trains exactly one rollout per micro-batch."
            )

        self.rank = int(rank)
        self.world_size = int(world_size)
        configured_gpu_ids = self.learner_cfg.get("gpu_ids", [])
        if not isinstance(configured_gpu_ids, list):
            raise ValueError("learner.gpu_ids must be a list of GPU ids.")
        if gpu_id is None:
            gpu_id = int(configured_gpu_ids[rank]) if rank < len(configured_gpu_ids) else rank
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer.")
        if configured_gpu_ids and gpu_id not in configured_gpu_ids:
            raise ValueError("gpu_id must be one of learner.gpu_ids.")
        if gpu_gate is not None and gpu_gate.physical_gpu_id != gpu_id:
            raise ValueError("Learner GPU gate must match the Learner physical gpu_id.")
        self.gpu_id = gpu_id
        self.gpu_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=gpu_id)
        self.device = torch.device(device) if device is not None else self._default_device()
        self.plugin: AlgorithmPlugin | None = None
        self.tokenizer: object | None = None
        self.model: torch.nn.Module | None = None
        self.model_engine: object | None = None
        self.optimizer: object | None = None
        self.lr_scheduler: object | None = None
        self.lora_enabled = bool(_mapping(self.policy_cfg.get("lora", {}), "run.policy.lora").get("enabled", False))
        if plugin is not None:
            self.set_plugin(plugin)
        self.global_step = 0
        self._initialized = False
        self._model_on_gpu = False
        self._model_load_count = 0
        self._model_offload_count = 0

        self.forward_chunk_size = int(self.execution_cfg.get("forward_chunk_size", 512))
        if self.forward_chunk_size <= 0:
            raise ValueError("learner.execution.forward_chunk_size must be positive.")
        self.max_batches_per_turn = int(self.execution_cfg.get("max_batches_per_turn", 1))
        if self.max_batches_per_turn < 1:
            raise ValueError("learner.execution.max_batches_per_turn must be positive.")
        self.clip_grad_norm = float(self.execution_cfg.get("clip_grad_norm", 0.0) or 0.0)
        if not math.isfinite(self.clip_grad_norm) or self.clip_grad_norm < 0:
            raise ValueError("learner.execution.clip_grad_norm must be a finite non-negative number.")
        self.max_sequence_length = int(self.learner_cfg.get("max_sequence_length", 0))
        if self.max_sequence_length <= 0:
            raise ValueError("learner.max_sequence_length must be positive.")
        if self.cpu_offload and self.world_size != 1:
            raise ValueError("Learner cpu_offload currently supports exactly one Learner rank.")
        if self.cpu_offload:
            zero = self.deepspeed_config().get("zero_optimization", {})
            if not isinstance(zero, Mapping) or int(zero.get("stage", 0)) != 0:
                raise ValueError("Learner cpu_offload currently requires DeepSpeed ZeRO stage 0.")

    def _default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        return torch.device("cpu")

    def set_plugin(self, plugin: AlgorithmPlugin) -> None:
        if not isinstance(plugin, AlgorithmPlugin):
            raise TypeError("plugin must inherit AlgorithmPlugin.")
        if plugin.requires_frozen_reference and not self.lora_enabled:
            raise ValueError(f"{type(plugin).__name__} requires LoRA to keep its reference Teacher fixed.")
        self.plugin = plugin

    def queue_keys(self) -> dict[str, str]:
        name, prefix = _run_name_and_redis(self.config)
        root = f"{prefix}:{name}"
        return {
            "learner_train": f"{root}:{self.node_id}:train",
            "controller_train_results": f"{root}:controller:train_results",
            "controller_events": f"{root}:controller:events",
        }

    def initialize_engine(self) -> None:
        """Load tokenizer/model/LoRA and wrap trainable parameters in DeepSpeed."""

        if self._initialized:
            return
        self._validate_runtime_memory_before_load()
        model_path = os.path.expanduser(str(self.policy_cfg.get("model_name_or_path", "")))
        if not model_path:
            raise ValueError("run.policy.model_name_or_path must be configured.")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - runtime dependency failure.
            raise RuntimeError("transformers is required to initialize the Learner.") from error

        dtype = _torch_dtype(self.policy_cfg.get("dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            # Packed answer probes require a model path that accepts custom
            # 4-D masks; FlashAttention2 does not provide that contract.
            "attn_implementation": "sdpa",
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.config.use_cache = False
        self._force_mask_compatible_attention(self.model)
        if bool(self.execution_cfg.get("gradient_checkpointing", True)):
            enable = getattr(self.model, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()

        self.model = self._attach_or_restore_lora(self.model)
        self._validate_model_load_headroom()
        self.model.to(self.device)
        self._model_on_gpu = self.device.type == "cuda"
        self.optimizer = self._build_optimizer(self.model)
        configured_scheduler = self._build_scheduler(self.optimizer)
        self.model_engine, self.optimizer, _unused, engine_scheduler = self._initialize_deepspeed(
            self.model,
            self.optimizer,
            lr_scheduler=configured_scheduler,
        )
        self.lr_scheduler = engine_scheduler if engine_scheduler is not None else configured_scheduler
        self._validate_runtime_memory_after_load()
        if self.cpu_offload:
            self._offload_model()
        self._initialized = True

    @staticmethod
    def _force_mask_compatible_attention(model: torch.nn.Module) -> None:
        setter = getattr(model, "set_attn_implementation", None)
        if callable(setter):
            setter("sdpa")
            return
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            setattr(config, "_attn_implementation", "sdpa")

    def _attach_or_restore_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.lora_enabled:
            return model
        try:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        except ImportError as error:  # pragma: no cover - runtime dependency failure.
            raise RuntimeError("peft is required when run.policy.lora.enabled=true.") from error
        lora_cfg = _mapping(self.policy_cfg.get("lora", {}), "run.policy.lora")
        step, adapter = _latest_lora_adapter(self.config)
        if adapter is not None:
            try:
                restored = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
            except Exception as error:
                # A partially copied or incompatible adapter must not prevent
                # a fresh run.  The actor only observes atomically published
                # directories, but an old manual checkpoint can still fail to
                # load after a model or PEFT-version change.
                print(
                    f"[Learner] Could not restore LoRA adapter {adapter}: {error}. "
                    "Starting from a fresh adapter.",
                    flush=True,
                )
            else:
                self.global_step = max(self.global_step, step)
                return restored
        targets = lora_cfg.get("target_modules", [])
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise TypeError("run.policy.lora.target_modules must be a sequence of module names.")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("rank", 64)),
            lora_alpha=int(lora_cfg.get("alpha", 128)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=[str(name) for name in targets],
            inference_mode=False,
        )
        return get_peft_model(model, config)

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        raw = self.learner_cfg.get("optimizer", {})
        cfg = _mapping(raw, "learner.optimizer") if raw else {}
        name = str(cfg.get("type", "adamw")).strip().lower()
        learning_rate = float(cfg.get("lr", 1e-5))
        if learning_rate <= 0:
            raise ValueError("learner.optimizer.lr must be positive.")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Learner model has no trainable parameters.")
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=tuple(cfg.get("betas", (0.9, 0.999))),
                eps=float(cfg.get("eps", 1e-8)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
            )
        if name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=tuple(cfg.get("betas", (0.9, 0.999))),
                eps=float(cfg.get("eps", 1e-8)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
            )
        raise ValueError("learner.optimizer.type must be 'adamw' or 'adam'.")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> object | None:
        """Build the small generic scheduler surface supported by Learner.

        Algorithm code does not create or step schedulers.  The framework
        supports a stable constant default plus linear and cosine decay for
        configurations that request them.  Non-constant schedules derive
        their horizon from ``controller.max_steps`` unless ``total_steps`` is
        specified explicitly under ``learner.scheduler``.
        """

        raw = self.learner_cfg.get("scheduler", {})
        cfg = _mapping(raw, "learner.scheduler") if raw else {}
        kind = str(cfg.get("type", "constant")).strip().lower()
        if kind == "constant":
            return None
        if kind not in {"linear", "cosine"}:
            raise ValueError("learner.scheduler.type must be 'constant', 'linear', or 'cosine'.")

        warmup_steps = int(cfg.get("warmup_steps", 0))
        if warmup_steps < 0:
            raise ValueError("learner.scheduler.warmup_steps may not be negative.")
        total_steps_value = cfg.get("total_steps")
        if total_steps_value is None:
            controller = _mapping(self.config.get("controller", {}), "controller")
            total_steps_value = controller.get("max_steps")
        total_steps = int(total_steps_value or 0)
        if total_steps <= 0:
            raise ValueError(
                "A linear or cosine learner.scheduler requires a positive total_steps "
                "or controller.max_steps."
            )
        if warmup_steps >= total_steps:
            raise ValueError("learner.scheduler.warmup_steps must be smaller than total_steps.")
        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("learner.scheduler.min_lr_ratio must lie in [0, 1].")

        def multiplier(step: int) -> float:
            # LambdaLR evaluates step zero during construction, before the
            # first optimizer update.  That makes warmup's first update use
            # 1 / warmup_steps of the configured learning rate.
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(1.0, (step - warmup_steps) / float(total_steps - warmup_steps))
            if kind == "linear":
                decay = 1.0 - progress
            else:
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)

    def deepspeed_config(self) -> dict[str, object]:
        """Build the stage-zero configuration used by this lightweight Learner."""

        dtype = _torch_dtype(self.policy_cfg.get("dtype", "bfloat16"))
        raw = self.learner_cfg.get("deepspeed", {})
        configured = _mapping(raw, "learner.deepspeed") if raw else {}
        configured.pop("gradient_clipping", None)
        if "train_micro_batch_size_per_gpu" in configured:
            raise ValueError(
                "learner.deepspeed.train_micro_batch_size_per_gpu is fixed at 1 and must not be configured."
            )
        result: dict[str, object] = {
            # FrugalOPL deliberately trains one rollout at a time.  This is
            # both the physical model-forward shape and the DeepSpeed batch
            # contract; logical batches are accumulated explicitly below.
            "train_micro_batch_size_per_gpu": 1,
            # Learner performs one optimizer step after the logical batch.
            # It accumulates individual micro-batch backwards explicitly.
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
            # FrugalOPL owns clipping through learner.execution.clip_grad_norm
            # immediately before the one logical-batch optimizer step.  Keep
            # DeepSpeed's own setting disabled so it cannot clip twice.
            "gradient_clipping": 0.0,
            "bf16": {"enabled": dtype == torch.bfloat16 and self.device.type == "cuda"},
            "fp16": {"enabled": dtype == torch.float16 and self.device.type == "cuda"},
        }
        for key, value in configured.items():
            if key in {"bf16", "fp16", "zero_optimization"} and isinstance(value, Mapping):
                nested = dict(result.get(key, {}))
                nested.update(copy.deepcopy(dict(value)))
                result[key] = nested
            else:
                result[key] = copy.deepcopy(value)
        return result

    def _initialize_deepspeed(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        lr_scheduler: object | None = None,
    ) -> tuple[object, object, object, object]:
        try:
            import deepspeed
        except ImportError as error:  # pragma: no cover - setup failure.
            raise RuntimeError("DeepSpeed is required to initialize the Learner.") from error
        # DeepSpeed's optimizer wrapper assumes a process group even for one
        # rank.  Establish it from the explicit framework configuration
        # rather than relying on launcher-only MASTER_* environment variables.
        self.initialize_distributed()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return deepspeed.initialize(
            model=model,
            model_parameters=trainable,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.deepspeed_config(),
            dist_init_required=not dist.is_initialized(),
        )

    def _validate_runtime_memory_before_load(self) -> None:
        if self.device.type != "cuda":
            return
        minimum = float(self.runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        if minimum < 3:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb may not be below 3 GB.")
        free, _total = self._gpu_memory_info()
        if free < int(minimum * 1024**3):
            raise MemoryError(
                f"Refusing to initialize Learner: GPU has only {free / 1024**3:.2f} GB free; "
                f"the required safety floor is {minimum:.2f} GB."
            )

    def _validate_runtime_memory_after_load(self) -> None:
        if self.device.type != "cuda":
            return
        minimum = float(self.runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        free, _total = self._gpu_memory_info()
        if free < int(minimum * 1024**3):
            raise MemoryError(
                f"Learner initialization left only {free / 1024**3:.2f} GB free; "
                f"the required safety floor is {minimum:.2f} GB."
            )

    def _gpu_memory_info(self) -> tuple[int, int]:
        physical = os.environ.get("FRUGALOPL_PHYSICAL_GPU_ID", "0")
        command = [
            "nvidia-smi",
            "--id",
            str(physical),
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi response: {completed.stdout!r}")
        free_mib, total_mib = (int(field) for field in fields)
        return free_mib * 1024**2, total_mib * 1024**2

    def _module(self) -> torch.nn.Module:
        if self.model_engine is not None:
            module = getattr(self.model_engine, "module", None)
            if isinstance(module, torch.nn.Module):
                return module
        if isinstance(self.model, torch.nn.Module):
            return self.model
        raise RuntimeError("Learner model is not initialized.")

    def _model_size_bytes(self) -> int:
        module = self._module()
        return sum(item.numel() * item.element_size() for item in (*module.parameters(), *module.buffers()))

    def _activate_model(self) -> float:
        if not self.cpu_offload or self.device.type != "cuda" or not torch.cuda.is_available() or self._model_on_gpu:
            return 0.0
        self._validate_model_load_headroom()
        started = time.perf_counter()
        self._module().to(self.device)
        self._synchronize_cuda()
        self._model_on_gpu = True
        self._validate_runtime_memory_after_load()
        self._model_load_count += 1
        return time.perf_counter() - started

    def _offload_model(self) -> float:
        if not self.cpu_offload or self.device.type != "cuda" or not torch.cuda.is_available() or not self._model_on_gpu:
            return 0.0
        self._synchronize_cuda()
        started = time.perf_counter()
        self._zero_grad()
        self._module().to("cpu")
        self._synchronize_cuda()
        torch.cuda.empty_cache()
        self._model_on_gpu = False
        self._model_offload_count += 1
        return time.perf_counter() - started

    def _validate_model_load_headroom(self) -> None:
        """Keep the device-wide floor intact before any CPU-resident model moves in."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        minimum_gb = float(self.runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        if minimum_gb < 3:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb may not be below 3 GB.")
        free, _total = self._gpu_memory_info()
        required = int(minimum_gb * 1024**3) + self._model_size_bytes()
        if free < required:
            raise MemoryError(
                f"Learner model load needs {required / 1024**3:.2f} GB free, "
                f"but GPU has {free / 1024**3:.2f} GB."
            )

    def _require_tokenizer(self) -> object:
        if self.tokenizer is None:
            raise RuntimeError("Learner tokenizer is not initialized.")
        return self.tokenizer

    def _normalize_logical_signals(self, signals: Mapping[str, object]) -> Signals:
        """Expose Actor trajectory fields directly while preserving ``traj``."""

        source: Signals = copy.deepcopy(dict(signals))
        traj = source.get("traj")
        if not isinstance(traj, Mapping):
            raise ValueError("Learner input must contain a trajectory mapping at signals['traj'].")
        trajectory = dict(traj)
        # Token IDs and the rollout-to-prompt mapping are indispensable for
        # any policy update.  Decoded response text and truncation flags are
        # intentionally optional: only the framework answer-probe helper
        # needs them, and Actor plugins remain free to avoid returning them
        # for algorithms that do not use that helper.
        for field in ("prompt_ids", "response_ids", "prompt_indices"):
            if field not in trajectory:
                raise ValueError(f"Learner trajectory is missing required field traj['{field}'].")
        response_ids = _as_ragged_token_list(trajectory["response_ids"], "traj.response_ids")
        prompt_ids = _as_ragged_token_list(trajectory["prompt_ids"], "traj.prompt_ids")
        rollout_count = len(response_ids)
        if rollout_count == 0:
            raise ValueError("Learner input contains no rollout trajectories.")
        if len(prompt_ids) != rollout_count:
            raise ValueError("traj.prompt_ids and traj.response_ids must have the same rollout length.")
        if any(row.numel() == 0 for row in prompt_ids):
            raise ValueError("Every rollout prompt must contain at least one token.")
        if any(row.numel() == 0 for row in response_ids):
            raise ValueError("Every rollout response must contain at least one token.")
        prompt_indices = _tensor_to_python_int_list(
            trajectory["prompt_indices"], length=rollout_count, path="traj.prompt_indices"
        )
        responses: list[str] | None = None
        if "responses" in trajectory:
            raw_responses = trajectory["responses"]
            if not isinstance(raw_responses, Sequence) or isinstance(raw_responses, (str, bytes)):
                raise TypeError("traj.responses must be a sequence of strings.")
            if len(raw_responses) != rollout_count or not all(isinstance(item, str) for item in raw_responses):
                raise ValueError("traj.responses must be a list[str] aligned with traj.response_ids.")
            responses = list(raw_responses)
        truncated: list[bool] | None = None
        if "response_truncated" in trajectory:
            truncated = _tensor_to_python_bool_list(
                trajectory["response_truncated"], length=rollout_count, path="traj.response_truncated"
            )
        questions = source.get("questions")
        answers = source.get("answers")
        if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
            raise TypeError("Learner signals['questions'] must be a sequence.")
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            raise TypeError("Learner signals['answers'] must be a sequence.")
        if len(questions) != len(answers):
            raise ValueError("signals['questions'] and signals['answers'] must have the same length.")
        if any(index < 0 or index >= len(answers) for index in prompt_indices):
            raise ValueError("traj.prompt_indices contains an index outside signals['answers'].")

        source["traj"] = trajectory
        # A Learner plugin sees the generic rollout fields directly.  Keep the
        # nested ``traj`` too for downstream traceability, but do not force
        # algorithm code to drill through it for every token-aligned signal.
        for field in _ROLLOUT_CORE_FIELDS:
            if field in trajectory:
                source[field] = trajectory[field]
        if responses is not None:
            source["responses"] = responses
        source["prompt_ids"] = prompt_ids
        source["response_ids"] = response_ids
        source["prompt_indices"] = torch.tensor(prompt_indices, dtype=torch.long)
        if truncated is not None:
            source["response_truncated"] = torch.tensor(truncated, dtype=torch.bool)
        source["rollout_count"] = rollout_count
        return source

    def _plugin_answer_probe(
        self,
        batch: Signals,
        *,
        probability_difference_threshold: float,
    ) -> Signals:
        """Implement the complete-batch answer-probe capability for a plugin."""

        threshold = float(probability_difference_threshold)
        if not math.isfinite(threshold):
            raise ValueError("probability_difference_threshold must be finite.")
        logical = self._normalize_logical_signals(batch)
        response_ids = _as_ragged_token_list(logical["response_ids"], "response_ids")
        prompt_ids = _as_ragged_token_list(logical["prompt_ids"], "prompt_ids")
        if "responses" not in logical or "response_truncated" not in logical:
            raise ValueError(
                "answer_probe() requires Actor trajectory fields 'responses' and "
                "'response_truncated'. Request both in the Actor plugin_hook."
            )
        responses = list(logical["responses"])
        truncated = _tensor_to_python_bool_list(
            logical["response_truncated"], length=len(response_ids), path="response_truncated"
        )
        prompt_indices = _tensor_to_python_int_list(
            logical["prompt_indices"], length=len(response_ids), path="prompt_indices"
        )
        answers = list(logical["answers"])

        rewards = torch.zeros((len(response_ids),), dtype=torch.float32)
        probabilities = torch.full((len(response_ids), 2), -1.0, dtype=torch.float32)
        to_probe: list[_ProbePlan] = []
        tokenizer = self._require_tokenizer()

        for index, is_truncated in enumerate(truncated):
            answer = answers[prompt_indices[index]]
            if not is_truncated:
                rewards[index] = float(verify_response_answer(responses[index], str(answer)))
                continue
            try:
                probe = _build_answer_probe(tokenizer, answer)
                plan = _ProbePlan(
                    rollout_index=index,
                    prompt_ids=prompt_ids[index],
                    response_ids=response_ids[index],
                    probe=probe,
                )
                total_probe_length = int(
                    plan.prompt_ids.numel()
                    + plan.response_ids.numel()
                    + 2 * len(plan.probe.token_ids)
                )
                if total_probe_length > self.max_sequence_length:
                    # The original rollout exceeded the Learner's declared
                    # policy context budget; treat it like an unrescuable
                    # truncation rather than changing the rollout signal.
                    continue
                to_probe.append(plan)
            except Exception:
                # A failed optional rescue is a normal incorrect truncation.
                # Do not terminate an algorithm run or manufacture a reward.
                continue

        local_results: list[tuple[int, float, float]] = []
        local_indices = _balanced_indices(len(to_probe), self.world_size, self.rank)
        # A dual-probe mask grows quadratically with sequence length.  It is
        # deliberately one rollout wide, matching the fixed training shape
        # and eliminating any ragged cross-rollout padding.
        for position in local_indices:
            result = self._probe_plan_with_fallback(to_probe[position])
            if result is not None:
                local_results.append(result)

        gathered: list[list[tuple[int, float, float]] | None]
        if _distributed_active():
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, local_results)
        else:
            gathered = [local_results]

        # Every rank receives the same indexed results.  The explicit index
        # restore is what makes output order independent of multi-GPU timing.
        indexed: dict[int, tuple[float, float]] = {}
        for rank_results in gathered:
            for index, start_probability, end_probability in rank_results or []:
                indexed[int(index)] = (float(start_probability), float(end_probability))
        for index, (start_probability, end_probability) in indexed.items():
            if not (math.isfinite(start_probability) and math.isfinite(end_probability)):
                continue
            probabilities[index] = torch.tensor([start_probability, end_probability], dtype=torch.float32)
            rewards[index] = float(end_probability - start_probability > threshold)
        return {"rewards": rewards, "answer_probe_probs": probabilities}

    def _probe_plan_with_fallback(self, plan: _ProbePlan) -> tuple[int, float, float] | None:
        """Run one answer probe, degrading a malformed plan to reward zero."""

        try:
            return self._run_probe_plan(plan)
        except Exception:
            # Answer probing is an optional rescue for truncated reasoning.
            # A single failure must preserve the normal zero-reward outcome
            # without affecting any other rollout.
            return None

    def _run_probe_plan(self, plan: _ProbePlan) -> tuple[int, float, float]:
        """Score one rollout's head/tail gold-answer branches in one forward."""

        module = self._module()
        length = int(plan.prompt_ids.numel() + plan.response_ids.numel() + 2 * len(plan.probe.token_ids))
        if length <= 1:
            raise ValueError("Answer-probe sequences require at least two tokens.")
        configured_limit = int(self.max_sequence_length)
        if length > configured_limit:
            raise ValueError("Answer-probe sequence exceeds learner.max_sequence_length.")
        model_limit = getattr(getattr(module, "config", None), "max_position_embeddings", None)
        if model_limit is not None and length > int(model_limit):
            raise ValueError("Answer-probe sequence exceeds the model position limit.")
        input_ids = torch.empty((1, length), dtype=torch.long, device=self.device)
        position_ids = torch.empty((1, length), dtype=torch.long, device=self.device)
        mask_dtype = self._module_mask_dtype(module)
        blocked = torch.finfo(mask_dtype).min
        # A single dense [1, 1, L, L] mask is necessary for arbitrary branch
        # visibility in stock SDPA.  With no cross-rollout batch dimension,
        # it has no padding rows and stays bounded by one trajectory.
        attention_mask = torch.full(
            (1, 1, length, length),
            blocked,
            dtype=mask_dtype,
            device=self.device,
        )
        prompt_length = int(plan.prompt_ids.numel())
        response_length = int(plan.response_ids.numel())
        probe_length = len(plan.probe.token_ids)
        original_length = prompt_length + response_length
        head_start = original_length
        head_end = head_start + probe_length
        tail_start = head_end
        tail_end = tail_start + probe_length
        if tail_end != length:  # pragma: no cover - defensive invariant.
            raise RuntimeError("Answer-probe sequence length mismatch.")

        input_ids[0, :prompt_length] = plan.prompt_ids.to(self.device, dtype=torch.long)
        input_ids[0, prompt_length:original_length] = plan.response_ids.to(self.device, dtype=torch.long)
        probe_ids = torch.tensor(plan.probe.token_ids, dtype=torch.long, device=self.device)
        input_ids[0, head_start:head_end] = probe_ids
        input_ids[0, tail_start:tail_end] = probe_ids

        # The head branch logically follows only P; the tail branch logically
        # follows P + R.  These rebased positions prevent both response
        # length and the first gold-answer branch leaking through RoPE.
        position_ids[0, :original_length] = torch.arange(original_length, device=self.device)
        position_ids[0, head_start:head_end] = torch.arange(
            prompt_length, prompt_length + probe_length, device=self.device
        )
        position_ids[0, tail_start:tail_end] = torch.arange(
            original_length, original_length + probe_length, device=self.device
        )

        # Start from ordinary causal visibility, then remove the two
        # forbidden physical prefixes.  This gives:
        #   head -> P + causal(head)
        #   tail -> P + R + causal(tail)
        local_mask = attention_mask[0, 0]
        local_mask.triu_(diagonal=1)
        local_mask[head_start:head_end, prompt_length:original_length] = blocked
        local_mask[tail_start:tail_end, head_start:head_end] = blocked

        offsets = torch.tensor(plan.probe.answer_token_positions, device=self.device, dtype=torch.long)
        head_targets = offsets + head_start
        tail_targets = offsets + tail_start
        target_ids = probe_ids.index_select(0, offsets)

        previous_mode = bool(module.training)
        try:
            module.eval()
            with torch.no_grad():
                with self._bounded_sdpa_context():
                    hidden = self._forward_hidden(
                        module,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                lm_head = self._lm_head(module)
                head_logits = lm_head(hidden[0].index_select(0, head_targets - 1))
                tail_logits = lm_head(hidden[0].index_select(0, tail_targets - 1))
                head = self._mean_selected_probability(head_logits, target_ids)
                tail = self._mean_selected_probability(tail_logits, target_ids)
                return plan.rollout_index, head, tail
        finally:
            if previous_mode:
                module.train()

    @staticmethod
    def _mean_selected_probability(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
        selected = logits.float().gather(1, target_ids.unsqueeze(1)).squeeze(1)
        probabilities = torch.exp(selected - torch.logsumexp(logits.float(), dim=-1))
        value = float(probabilities.mean().detach().cpu().item())
        if not math.isfinite(value):
            raise ValueError("Answer probe produced a non-finite probability.")
        return value

    @staticmethod
    def _module_mask_dtype(module: torch.nn.Module) -> torch.dtype:
        for parameter in module.parameters():
            if parameter.dtype.is_floating_point:
                return parameter.dtype
        return torch.float32

    def _bounded_sdpa_context(self) -> object:
        """Forbid unsafe math attention in CUDA probes and training.

        A 4-D branch mask is small enough by itself at B=1, but a math SDPA
        fallback can materialize one attention-score matrix per head.  On CUDA
        we therefore accept only Flash or memory-efficient SDPA kernels.  If
        neither accepts the particular mask, the optional probe cleanly falls
        back to reward zero. Training keeps this context active through
        backward so checkpoint recomputation follows the same memory bound.
        """

        if self.device.type != "cuda":
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        except (ImportError, AttributeError) as error:  # pragma: no cover - older torch.
            raise RuntimeError("CUDA Learner requires Flash or memory-efficient SDPA.") from error

    def _forward_hidden(
        self,
        module: torch.nn.Module,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder backbone without materializing full vocabulary logits."""

        backbone = self._backbone(module)
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Learner backbone did not return last_hidden_state.")
        return hidden

    @staticmethod
    def _backbone(module: torch.nn.Module) -> Callable[..., object]:
        base_getter = getattr(module, "get_base_model", None)
        base = base_getter() if callable(base_getter) else None
        candidate = getattr(base, "model", None)
        if callable(candidate):
            return candidate
        candidate = getattr(module, "model", None)
        if callable(candidate):
            return candidate
        raise RuntimeError("Learner model must expose a decoder backbone at .model.")

    @staticmethod
    def _lm_head(module: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
        base_getter = getattr(module, "get_base_model", None)
        base = base_getter() if callable(base_getter) else None
        output_embeddings = getattr(base, "get_output_embeddings", None)
        head = output_embeddings() if callable(output_embeddings) else None
        if callable(head):
            return head
        head = getattr(base, "lm_head", None)
        if callable(head):
            return head
        output_embeddings = getattr(module, "get_output_embeddings", None)
        head = output_embeddings() if callable(output_embeddings) else None
        if callable(head):
            return head
        head = getattr(module, "lm_head", None)
        if callable(head):
            return head
        raise RuntimeError("Learner model must expose an lm_head.")

    def _plugin_collect_logits(
        self,
        batch: Signals,
        *,
        mode: str,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
        return_signals: Mapping[str, object] | None,
    ) -> Signals:
        """Score a logical batch before training and restore ragged CPU order."""

        if mode not in {"reference", "policy_no_grad"}:
            raise ValueError("Learner signal collection permits only no-gradient compute_logits modes.")
        if prompts is not None and prompt_ids is not None:
            raise ValueError("Use either prompts or prompt_ids, not both.")
        logical = self._normalize_logical_signals(batch)
        total = int(logical["rollout_count"])
        required = _normalize_logit_signals(return_signals)
        overrides: list[torch.Tensor] | None = None
        if prompts is not None:
            if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
                raise TypeError("prompts must be a sequence of rendered prompt strings.")
            if len(prompts) != total:
                raise ValueError("Signal-hook prompts must contain one rendered prompt per rollout.")
        if prompt_ids is not None:
            overrides = _as_ragged_token_list(prompt_ids, "prompt_ids")
            if len(overrides) != total:
                raise ValueError("Signal-hook prompt_ids must contain one prompt per rollout.")

        local: list[tuple[int, dict[str, torch.Tensor]]] = []
        with self._bounded_sdpa_context():
            for index in _balanced_indices(total, self.world_size, self.rank):
                micro_batch = self._build_rollout_micro_batch(logical, index)
                computed = self._plugin_compute_logits(
                    micro_batch,
                    mode=mode,
                    prompts=[prompts[index]] if prompts is not None else None,
                    prompt_ids=[overrides[index]] if overrides is not None else None,
                    return_signals=required,
                )
                local.append((index, {
                    key: value.detach().cpu().squeeze(0).contiguous()
                    for key, value in computed.items()
                }))

        gathered: list[object | None]
        if _distributed_active():
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        ordered: list[dict[str, torch.Tensor] | None] = [None] * total
        for rank_results in gathered:
            for index, signals in rank_results or []:
                if not 0 <= index < total or ordered[index] is not None:
                    raise RuntimeError("Learner collected duplicate or invalid rollout indices.")
                ordered[index] = signals
        if any(signals is None for signals in ordered):
            raise RuntimeError("Learner signal collection did not return every rollout.")
        keys = tuple(ordered[0])
        if any(set(signals) != set(keys) for signals in ordered):
            raise RuntimeError("Learner signal collection returned inconsistent fields across rollouts.")
        return {key: [signals[key] for signals in ordered] for key in keys}

    def _plugin_compute_logits(
        self,
        batch: Signals,
        *,
        mode: str | None,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
        return_signals: Mapping[str, object] | None,
    ) -> Signals:
        """Implement the micro-batch token-signal capability for a plugin."""

        if mode not in {"reference", "policy_no_grad", "policy_grad"}:
            raise ValueError(f"Unsupported compute_logits mode: {mode!r}.")
        if prompts is not None and prompt_ids is not None:
            raise ValueError("Use either prompts or prompt_ids, not both.")
        module = self._module()
        required = _normalize_logit_signals(return_signals)
        prepared = self._prepare_forward_batch(batch, prompts=prompts, prompt_ids=prompt_ids)
        requires_grad = mode == "policy_grad"
        disable_lora = mode == "reference"
        return self._compute_token_signals(
            module,
            prepared,
            required,
            requires_grad=requires_grad,
            disable_lora=disable_lora,
        )

    def _prepare_forward_batch(
        self,
        batch: Mapping[str, object],
        *,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
    ) -> dict[str, torch.Tensor]:
        """Prepare one unpadded rollout plus an optional prompt override."""

        response_ids = batch.get("response_ids")
        response_mask = batch.get("response_mask")
        if not isinstance(response_ids, torch.Tensor) or response_ids.ndim != 2:
            raise TypeError("compute_logits() requires a rank-two batch['response_ids'] tensor.")
        if not isinstance(response_mask, torch.Tensor) or tuple(response_mask.shape) != tuple(response_ids.shape):
            raise TypeError("compute_logits() requires response_mask aligned with response_ids.")
        batch_size = int(response_ids.size(0))
        if batch_size != 1:
            raise ValueError(
                "compute_logits() requires exactly one rollout; "
                "FrugalOPL fixes the training micro-batch size at 1."
            )

        if prompts is not None:
            if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
                raise TypeError("prompts must be a sequence containing one rendered prompt string.")
            if len(prompts) != 1:
                raise ValueError("prompts must contain exactly one prompt for the one-rollout micro-batch.")
            prompt_tensor = self._encode_prompt(prompts[0]).unsqueeze(0)
            prompt_mask = torch.ones_like(prompt_tensor, dtype=torch.bool)
        elif prompt_ids is not None:
            prompt_list = _as_ragged_token_list(prompt_ids, "prompt_ids")
            if len(prompt_list) != 1:
                raise ValueError("prompt_ids must contain exactly one prompt for the one-rollout micro-batch.")
            prompt_tensor = prompt_list[0].unsqueeze(0)
            prompt_mask = torch.ones_like(prompt_tensor, dtype=torch.bool)
        else:
            prompt_tensor = batch.get("prompt_ids")
            prompt_mask = batch.get("prompt_attention_mask")
            if not isinstance(prompt_tensor, torch.Tensor) or prompt_tensor.ndim != 2:
                raise TypeError("compute_logits() requires a rank-two batch['prompt_ids'] tensor.")
            if not isinstance(prompt_mask, torch.Tensor) or tuple(prompt_mask.shape) != tuple(prompt_tensor.shape):
                raise TypeError("compute_logits() requires prompt_attention_mask aligned with prompt_ids.")
            if int(prompt_tensor.size(0)) != 1:
                raise ValueError("prompt_ids must contain exactly one rollout.")
            prompt_tensor = prompt_tensor.detach().to(device=self.device, dtype=torch.long)
            prompt_mask = prompt_mask.detach().to(device=self.device, dtype=torch.bool)

        if int(prompt_tensor.size(1)) < 1 or int(response_ids.size(1)) < 1:
            raise ValueError("prompt_ids and response_ids must both contain at least one sequence position.")
        if not bool(prompt_mask.all()):
            raise ValueError("compute_logits() receives one unpadded prompt; prompt_attention_mask must be all true.")
        if not bool(response_mask.to(dtype=torch.bool).all()):
            raise ValueError("compute_logits() receives one unpadded response; response_mask must be all true.")

        prompt_tensor = prompt_tensor.to(self.device, dtype=torch.long)
        prompt_mask = prompt_mask.to(self.device, dtype=torch.bool)
        response_tensor = response_ids.detach().to(self.device, dtype=torch.long)
        response_mask = response_mask.detach().to(self.device, dtype=torch.bool)
        input_ids = torch.cat([prompt_tensor, response_tensor], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        sequence_length = int(input_ids.size(1))
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "compute_logits() sequence length exceeds learner.max_sequence_length "
                f"({sequence_length} > {self.max_sequence_length})."
            )
        model_limit = getattr(getattr(self._module(), "config", None), "max_position_embeddings", None)
        if model_limit is not None and sequence_length > int(model_limit):
            raise ValueError(
                "compute_logits() sequence length exceeds the model position limit "
                f"({sequence_length} > {int(model_limit)})."
            )
        # Both masks are proven all-true above, so positions are simply the
        # contiguous sequence indices.  There is no padded-row adjustment in
        # the fixed single-rollout execution path.
        position_ids = torch.arange(sequence_length, device=self.device, dtype=torch.long).unsqueeze(0)
        return {
            "prompt_ids": prompt_tensor,
            "prompt_attention_mask": prompt_mask,
            "response_ids": response_tensor,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _encode_prompt(self, prompt: object) -> torch.Tensor:
        tokenizer = self._require_tokenizer()
        if not isinstance(prompt, str):
            raise TypeError("prompts must contain only strings.")
        tokenizer_call = getattr(tokenizer, "__call__", None)
        if not callable(tokenizer_call):
            raise TypeError("Learner tokenizer must be callable to encode prompt overrides.")
        encoded = tokenizer_call(prompt, add_special_tokens=False)
        if not isinstance(encoded, Mapping):
            raise TypeError("Tokenizer prompt output must be a mapping.")
        ids = _tokenizer_result_ids(encoded.get("input_ids"), "prompts[0]")
        if not ids:
            raise ValueError("Prompt overrides may not tokenize to an empty sequence.")
        return torch.tensor(ids, dtype=torch.long)

    def _compute_token_signals(
        self,
        module: torch.nn.Module,
        prepared: Mapping[str, torch.Tensor],
        required: Mapping[str, object],
        *,
        requires_grad: bool,
        disable_lora: bool,
    ) -> Signals:
        """Run one compact policy/reference forward and select requested tokens."""

        response_ids = prepared["response_ids"]
        prompt_length = int(prepared["prompt_ids"].size(1))
        response_length = int(response_ids.size(1))
        prior_training = bool(module.training)
        grad_context = nullcontext() if requires_grad else torch.no_grad()
        adapter_context = self._disable_lora(module) if disable_lora else nullcontext()
        try:
            if requires_grad:
                module.train()
            else:
                module.eval()
            with adapter_context:
                with grad_context:
                    hidden = self._forward_hidden(
                        module,
                        input_ids=prepared["input_ids"],
                        attention_mask=prepared["attention_mask"],
                        position_ids=prepared["position_ids"],
                    )
                    response_hidden = hidden[:, prompt_length - 1 : prompt_length - 1 + response_length, :]
                    return self._signals_from_response_hidden(
                        module,
                        response_hidden=response_hidden,
                        response_ids=response_ids,
                        required=required,
                    )
        finally:
            if prior_training:
                module.train()
            else:
                module.eval()

    @staticmethod
    def _disable_lora(module: torch.nn.Module) -> object:
        """Return PEFT's adapter-disable context, or a harmless no-op context."""

        for candidate in (module, getattr(module, "base_model", None)):
            disable = getattr(candidate, "disable_adapter", None)
            if callable(disable):
                return disable()
        return nullcontext()

    def _signals_from_response_hidden(
        self,
        module: torch.nn.Module,
        *,
        response_hidden: torch.Tensor,
        response_ids: torch.Tensor,
        required: Mapping[str, object],
    ) -> Signals:
        """Compute selected response-token statistics in bounded vocab chunks."""

        want_logprobs = bool(required.get("response_logprobs", True))
        want_entropy = bool(required.get("response_entropy", False))
        entropy_requires_grad = bool(required.get("entropy_requires_grad", True))
        topk_cfg = _mapping(required.get("topk", {}), "return_signals.topk")
        want_topk = bool(topk_cfg.get("enabled", False))
        top_k = int(topk_cfg.get("top_k", 0) or 0)
        if want_topk and top_k <= 0:
            raise ValueError("return_signals.topk.top_k must be positive when enabled.")

        outputs: dict[str, list[torch.Tensor]] = {
            "response_logprobs": [],
            "response_entropy": [],
            "topk_token_ids": [],
            "topk_logprobs": [],
        }
        lm_head = self._lm_head(module)
        length = int(response_ids.size(1))
        for start in range(0, length, self.forward_chunk_size):
            end = min(start + self.forward_chunk_size, length)
            hidden_chunk = response_hidden[:, start:end, :]
            ids_chunk = response_ids[:, start:end]
            logits = lm_head(hidden_chunk)
            float_logits = logits.float()
            normalization = torch.logsumexp(float_logits, dim=-1)
            if want_logprobs:
                selected = float_logits.gather(-1, ids_chunk.unsqueeze(-1)).squeeze(-1) - normalization
                outputs["response_logprobs"].append(selected)
            if want_entropy:
                # Logging-only entropy shares logits without retaining vocab-sized gradients.
                with torch.set_grad_enabled(torch.is_grad_enabled() and entropy_requires_grad):
                    log_probs = float_logits - normalization.unsqueeze(-1)
                    probabilities = torch.exp(log_probs)
                    entropy = -(probabilities * log_probs).sum(dim=-1)
                outputs["response_entropy"].append(entropy)
                del log_probs, probabilities
            if want_topk:
                values, token_ids = torch.topk(float_logits - normalization.unsqueeze(-1), k=top_k, dim=-1)
                token_ids = token_ids.to(dtype=torch.int32)
                outputs["topk_token_ids"].append(token_ids)
                outputs["topk_logprobs"].append(values)

        result: Signals = {}
        for key, chunks in outputs.items():
            if chunks:
                result[key] = torch.cat(chunks, dim=1)
        return result

    def build_micro_batches(self, signals: Mapping[str, object]) -> list[Signals]:
        """Build one unpadded ``[1, sequence_length]`` batch per rollout."""

        logical = self._normalize_logical_signals(signals)
        total = int(logical["rollout_count"])
        return [self._build_rollout_micro_batch(logical, rollout_index) for rollout_index in range(total)]

    def _build_rollout_micro_batch(self, logical: Mapping[str, object], rollout_index: int) -> Signals:
        """Materialize one rollout without collation or token padding."""

        total = int(logical.get("rollout_count", 0))
        if total < 1 or rollout_index < 0 or rollout_index >= total:
            raise ValueError("The rollout index is outside the logical rollout batch.")
        prompts = _as_ragged_token_list(logical.get("prompt_ids"), "prompt_ids")
        responses = _as_ragged_token_list(logical.get("response_ids"), "response_ids")
        prompt = prompts[rollout_index]
        response = responses[rollout_index]
        if prompt.numel() == 0:
            raise ValueError("Every rollout prompt must contain at least one token.")
        if response.numel() == 0:
            raise ValueError("Every rollout response must contain at least one token.")

        # The leading dimension remains explicit for every plugin, but it is
        # permanently one.  Each token belongs to this rollout, so no
        # attention or response padding masks need to be constructed.
        prompt_ids = prompt.to(dtype=torch.long).unsqueeze(0)
        response_ids = response.to(dtype=torch.long).unsqueeze(0)
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        response_mask = torch.ones_like(response_ids, dtype=torch.bool)

        result: Signals = {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "response_mask": response_mask,
            "rollout_indices": torch.tensor([rollout_index], dtype=torch.long),
        }
        for key, value in logical.items():
            if key in {
                "traj",
                "prompt_ids",
                "response_ids",
                "prompt_attention_mask",
                "response_mask",
                "input_ids",
                "attention_mask",
                "rollout_indices",
                "rollout_count",
            }:
                continue
            result[key] = self._select_rollout_signal_value(
                key,
                value,
                rollout_index=rollout_index,
                total=total,
            )
        result["traj"] = {
            key: result[key]
            for key in _ROLLOUT_CORE_FIELDS
            if key in result
        }
        return result

    def _select_rollout_signal_value(
        self,
        key: str,
        value: object,
        *,
        rollout_index: int,
        total: int,
    ) -> object:
        """Select one rollout and retain its leading batch dimension."""

        # Controller fields are prompt-level metadata even when P happens to
        # equal R, so never infer rollout alignment from their coincidental
        # length.
        if key in {"questions", "answers", "records", "actor_metrics"}:
            return copy.deepcopy(value)
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == total:
            return value.detach().cpu()[rollout_index : rollout_index + 1].contiguous()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != total:
            return copy.deepcopy(value)

        selected = value[rollout_index]
        if isinstance(selected, torch.Tensor):
            return selected.detach().cpu().unsqueeze(0).contiguous()
        if isinstance(selected, bool):
            return torch.tensor([selected], dtype=torch.bool)
        if isinstance(selected, int):
            return torch.tensor([selected], dtype=torch.long)
        if isinstance(selected, float):
            return torch.tensor([selected], dtype=torch.float32)
        # Text and arbitrary plugin metadata preserve one item per rollout
        # without pretending that they are tensor-shaped model inputs.
        return copy.deepcopy([selected])

    @staticmethod
    def _centralized_training_metrics(
        plugin_metrics: Mapping[str, object],
        learner_metrics: Mapping[str, object],
    ) -> tuple[Metrics, Metrics]:
        """Build Controller-owned Algorithm and Learner telemetry envelopes.

        Worker processes never write experiment telemetry directly.  The
        result sent over Redis instead carries two unambiguous components for
        the Controller's one SwanLab run:

        * ``algorithm_metrics`` retains every numeric plugin metric and
          exposes the optimizer-facing ``loss`` convention.
        * ``learner_metrics`` contains framework metrics and mirrors the
          logical-batch loss plus Learner-owned gradient telemetry needed to
          interpret an update independently of a particular AlgorithmPlugin.

        ``plugin_metrics`` and ``learner_metrics`` are already scalar-only
        mappings produced by :meth:`_train_micro_batches`; copies make the
        result independent of any later caller mutation.
        """

        algorithm: Metrics = {key: value for key, value in plugin_metrics.items() if key != "objective"}
        learner: Metrics = {key: value for key, value in learner_metrics.items() if key != "objective"}
        loss = _metric_number(algorithm.get("loss"))
        if loss is not None:
            learner["loss"] = loss

        raw_gradient = _metric_number(learner.get("raw_gradient"))
        if raw_gradient is None:
            # ``grad_norm`` predates the explicit raw/post-clip names.  Keep
            # accepting it from legacy Learner results while treating it as
            # the raw, pre-clip value.
            raw_gradient = _metric_number(learner.get("grad_norm"))
        if raw_gradient is not None:
            # ``_train_micro_batches`` samples this after all micro-batch
            # gradients have accumulated and before clipping.  It is useful
            # under both namespaces: Algorithm for optimization diagnostics,
            # Learner for framework execution diagnostics.  ``grad_norm``
            # remains a compatibility alias for ``raw_gradient``.
            algorithm["raw_gradient"] = raw_gradient
            learner["raw_gradient"] = raw_gradient
            algorithm["grad_norm"] = raw_gradient
            learner["grad_norm"] = raw_gradient

        clip_gradient = _metric_number(learner.get("clip_gradient"))
        if clip_gradient is not None:
            # This is sampled after Learner-owned clipping and immediately
            # before the optimizer step.  Algorithm plugins never need to
            # calculate or transport either gradient norm themselves.
            algorithm["clip_gradient"] = clip_gradient
            learner["clip_gradient"] = clip_gradient
        return algorithm, learner

    def process_training_batch(self, signals: Mapping[str, object]) -> Signals:
        """Run full-batch signal collection followed by one logical optimizer update."""

        if self.plugin is None:
            raise RuntimeError("Learner training requires an AlgorithmPlugin.")
        logical = self._normalize_logical_signals(signals)
        with _learner_plugin_scope(self, phase="collect"):
            additions = getattr(self.plugin, self.plugin_hook)(copy.deepcopy(logical))
        if not isinstance(additions, Mapping):
            raise TypeError(f"Learner plugin_hook {self.plugin_hook!r} must return a mapping of new signals.")
        if not all(isinstance(key, str) for key in additions):
            raise TypeError(f"Learner plugin_hook {self.plugin_hook!r} must return string signal names.")
        collisions = sorted(key for key in additions if key in logical)
        if collisions:
            raise ValueError(
                f"Learner plugin_hook {self.plugin_hook!r} may only return new fields; it cannot replace "
                f"an upstream signal: {collisions}."
            )
        enriched: Signals = dict(logical)
        enriched.update(dict(additions))
        micro_batches = self.build_micro_batches(enriched)
        metrics, raw_learner_metrics = self._train_micro_batches(micro_batches)
        algorithm_metrics, learner_metrics = self._centralized_training_metrics(
            metrics,
            raw_learner_metrics,
        )
        result: Signals = {
            "global_step": self.global_step,
            # ``metrics`` remains the legacy algorithm-metric name for
            # existing result consumers.  New Controller code should prefer
            # the explicit centralized ``algorithm_metrics`` envelope.
            "metrics": metrics,
            "algorithm_metrics": algorithm_metrics,
            "learner_metrics": learner_metrics,
        }
        # Preserve upstream Worker telemetry for the Controller's single
        # experiment run and for local JSON history.  All values are detached
        # below before the result crosses Redis.
        metric_keys = ["batch_id", "policy_version", "actor_metrics"]
        metric_keys.extend(f"{node.node_id}_metrics" for node in self.pipeline.teacher_nodes)
        for key in metric_keys:
            if key in logical:
                result[key] = copy.deepcopy(logical[key])
        if isinstance(logical.get("timing"), Mapping):
            # Framework timing is transport metadata, not an algorithm signal;
            # return it verbatim so Controller can persist the whole path.
            result["timing"] = copy.deepcopy(dict(logical["timing"]))
        return _deep_cpu_detach(result)  # type: ignore[return-value]

    def _train_micro_batches(self, micro_batches: Sequence[Signals]) -> tuple[Metrics, Metrics]:
        with self._bounded_sdpa_context():
            return self._train_micro_batches_impl(micro_batches)

    def _train_micro_batches_impl(self, micro_batches: Sequence[Signals]) -> tuple[Metrics, Metrics]:
        if not micro_batches:
            raise ValueError("A logical training batch must contain at least one micro-batch.")
        for batch in micro_batches:
            response_ids = batch.get("response_ids")
            if not isinstance(response_ids, torch.Tensor) or response_ids.ndim != 2:
                raise TypeError("Each physical training batch must contain a rank-two response_ids tensor.")
            if int(response_ids.size(0)) != 1:
                raise RuntimeError(
                    "Learner invariant violated: each physical training batch must contain exactly one rollout."
                )
        total_rollouts = len(micro_batches)
        slots = math.ceil(len(micro_batches) / self.world_size)
        metric_sums: dict[str, float] = {}
        metric_weights: dict[str, float] = {}
        objective_sum = 0.0
        objective_weight = 0.0
        self._zero_grad()

        for slot in range(slots):
            micro_index = slot * self.world_size + self.rank
            if micro_index < len(micro_batches):
                batch = _move_tensors_to_device(micro_batches[micro_index], self.device)
                if not isinstance(batch, dict):  # pragma: no cover - recursive helper invariant.
                    raise RuntimeError("Learner failed to materialize a micro-batch mapping.")
                with _learner_plugin_scope(self, phase="objective"):
                    objective, raw_metrics = getattr(self.plugin, self.objective_fn)(batch)  # type: ignore[union-attr]
                if not isinstance(objective, torch.Tensor) or objective.numel() != 1:
                    raise TypeError(f"Learner objective_fn {self.objective_fn!r} must return one scalar torch.Tensor objective.")
                if not objective.requires_grad:
                    raise ValueError(f"Learner objective_fn {self.objective_fn!r} must return a differentiable objective tensor.")
                if not isinstance(raw_metrics, Mapping):
                    raise TypeError(f"Learner objective_fn {self.objective_fn!r} metrics must be a mapping.")
                # Objectives have maximization semantics.  DDP/DeepSpeed
                # averages rank gradients, hence the world_size factor makes
                # one logical update equal to the mean across all rollouts.
                scale = self.world_size / float(total_rollouts)
                self._backward(-objective * scale)
                objective_sum += float(objective.detach().float().cpu().item())
                objective_weight += 1.0
                for key, value in raw_metrics.items():
                    name = str(key)
                    if name == "objective":
                        continue
                    numeric = _metric_number(value)
                    if numeric is None:
                        continue
                    metric_sums[name] = metric_sums.get(name, 0.0) + numeric
                    metric_weights[name] = metric_weights.get(name, 0.0) + 1.0
            else:
                # Every rank must participate in the same number of DDP
                # backwards.  A zero graph over trainable params provides the
                # required collective without affecting the logical gradient.
                self._backward(self._dummy_loss())

        # Capture the norm after every micro-batch contribution has been
        # accumulated and before optional clipping.  This is the only point
        # where a meaningful logical-batch raw gradient exists.  Keep the
        # ordering explicit: raw -> clip -> clipped -> optimizer step.
        raw_gradient = self._gradient_norm()
        self._clip_gradients()
        clip_gradient = self._gradient_norm()
        self._step()
        self.global_step += 1
        if _distributed_active():
            gathered: list[object | None] = [None for _ in range(self.world_size)]
            dist.all_gather_object(
                gathered,
                {
                    "objective_sum": objective_sum,
                    "objective_weight": objective_weight,
                    "metric_sums": metric_sums,
                    "metric_weights": metric_weights,
                },
            )
            objective_sum = 0.0
            objective_weight = 0.0
            metric_sums = {}
            metric_weights = {}
            for item in gathered:
                if not isinstance(item, Mapping):
                    raise RuntimeError("Learner metric gather returned an invalid payload.")
                objective_sum += float(item.get("objective_sum", 0.0))
                objective_weight += float(item.get("objective_weight", 0.0))
                remote_sums = item.get("metric_sums", {})
                remote_weights = item.get("metric_weights", {})
                if not isinstance(remote_sums, Mapping) or not isinstance(remote_weights, Mapping):
                    raise RuntimeError("Learner metric gather returned invalid metric mappings.")
                for key, value in remote_sums.items():
                    name = str(key)
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                for key, value in remote_weights.items():
                    name = str(key)
                    metric_weights[name] = metric_weights.get(name, 0.0) + float(value)
        metrics: Metrics = {
            key: metric_sums[key] / metric_weights[key]
            for key in sorted(metric_sums)
            if metric_weights[key] > 0
        }
        metrics["loss"] = -(objective_sum / objective_weight) if objective_weight > 0 else 0.0
        adapter_path = self._maybe_publish_adapter()
        learner_metrics: Metrics = {
            "train_rollouts": float(total_rollouts),
            "micro_batches": float(len(micro_batches)),
            "lr": self._current_lr(),
            # ``grad_norm`` remains the public compatibility spelling for
            # the pre-clip global L2 norm.  New consumers should use the
            # explicit raw/post-clip fields.
            "raw_gradient": raw_gradient,
            "clip_gradient": clip_gradient,
            "grad_norm": raw_gradient,
        }
        if adapter_path:
            learner_metrics["lora_path"] = adapter_path
        return metrics, learner_metrics

    def _zero_grad(self) -> None:
        engine_optimizer = getattr(self.model_engine, "optimizer", None)
        optimizer = engine_optimizer if engine_optimizer is not None else self.optimizer
        zero_grad = getattr(optimizer, "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except TypeError:
                zero_grad()
            return
        self._module().zero_grad(set_to_none=True)

    def _backward(self, loss: torch.Tensor) -> None:
        if self.model_engine is not None:
            backward = getattr(self.model_engine, "backward", None)
            if callable(backward):
                backward(loss)
                return
        loss.backward()

    def _step(self) -> None:
        if self.model_engine is not None:
            step = getattr(self.model_engine, "step", None)
            if callable(step):
                step()
                return
        if self.optimizer is None:
            raise RuntimeError("Learner optimizer is not initialized.")
        step = getattr(self.optimizer, "step", None)
        if not callable(step):
            raise RuntimeError("Learner optimizer does not provide step().")
        step()
        if self.lr_scheduler is not None:
            scheduler_step = getattr(self.lr_scheduler, "step", None)
            if callable(scheduler_step):
                scheduler_step()

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        parameters = [parameter for parameter in self._module().parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("Learner has no trainable parameters.")
        return parameters

    def _gradient_norm(self) -> float:
        """Return the pre-clip global L2 norm of trainable parameter gradients."""

        squared_norm = torch.zeros((), dtype=torch.float64, device=self.device)
        for parameter in self._trainable_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            value = torch.linalg.vector_norm(gradient.detach().float(), ord=2)
            squared_norm.add_(value.to(dtype=torch.float64).square())
        return float(torch.sqrt(squared_norm).detach().cpu().item())

    def _clip_gradients(self) -> None:
        """Clip once after the full logical-batch accumulation, when requested."""

        if self.clip_grad_norm <= 0:
            return
        torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm=self.clip_grad_norm)

    def _dummy_loss(self) -> torch.Tensor:
        trainable = [parameter for parameter in self._module().parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Learner has no trainable parameters for a distributed dummy backward.")
        return torch.stack([parameter.reshape(-1)[0] * 0.0 for parameter in trainable]).sum()

    def _current_lr(self) -> float:
        optimizer = getattr(self.model_engine, "optimizer", None) if self.model_engine is not None else None
        optimizer = optimizer or self.optimizer
        groups = getattr(optimizer, "param_groups", None)
        if isinstance(groups, Sequence) and groups:
            value = groups[0].get("lr") if isinstance(groups[0], Mapping) else None
            if value is not None:
                return float(value)
        return 0.0

    def _maybe_publish_adapter(self) -> str:
        if not self.lora_enabled or self.rank != 0:
            return ""
        save = getattr(self._module(), "save_pretrained", None)
        if not callable(save):
            return ""
        root = _lora_adapter_root(self.config)
        root.mkdir(parents=True, exist_ok=True)
        final_dir = root / f"step_{self.global_step}"
        adapter_dir = final_dir / "adapter_lora"
        if adapter_dir.is_dir():
            return str(adapter_dir)
        temporary_dir = root / f"step_{self.global_step}_tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_adapter = temporary_dir / "adapter_lora"
        temporary_adapter.parent.mkdir(parents=True, exist_ok=False)
        save(str(temporary_adapter))
        state_dir = temporary_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, object] = {"global_step": self.global_step}
        optimizer = getattr(self.model_engine, "optimizer", None) if self.model_engine is not None else None
        optimizer = optimizer or self.optimizer
        state_dict = getattr(optimizer, "state_dict", None)
        if callable(state_dict):
            state["optimizer"] = state_dict()
        scheduler_state = getattr(self.lr_scheduler, "state_dict", None)
        if callable(scheduler_state):
            state["scheduler"] = scheduler_state()
        torch.save(state, state_dir / "trainer_state.pt")
        os.replace(temporary_dir, final_dir)
        self._prune_old_adapters(root)
        return str(adapter_dir)

    def _prune_old_adapters(self, root: Path) -> None:
        max_keep = int(self.checkpoint_cfg.get("max_keep", 4))
        if max_keep < 1:
            raise ValueError("learner.checkpoint.max_keep must be positive.")
        candidates: list[tuple[int, Path]] = []
        for child in root.iterdir():
            match = re.fullmatch(r"step_(\d+)", child.name)
            if child.is_dir() and match and (child / "adapter_lora").is_dir():
                candidates.append((int(match.group(1)), child))
        candidates.sort()
        for _step, stale in candidates[:-max_keep]:
            shutil.rmtree(stale)

    def initialize_distributed(self) -> None:
        """Initialize this Learner's explicit process group, including rank zero alone."""

        if dist.is_initialized():
            return
        backend = str(self.distributed_cfg.get("backend", "nccl"))
        address = str(self.distributed_cfg.get("master_addr", "127.0.0.1"))
        port = int(self.distributed_cfg.get("master_port", 29573))
        timeout_seconds = int(self.distributed_cfg.get("timeout_sec", 7200))
        if self.device.type != "cuda" and backend.lower() == "nccl":
            raise ValueError("learner.distributed.backend='nccl' requires a CUDA Learner device.")
        # ``LearnerManager`` sets these before spawning a rank, but direct
        # single-rank use (including the real local smoke test) intentionally
        # bypasses that manager.  DeepSpeed still requires LOCAL_RANK even
        # when the framework supplies an explicit ``init_method``.  Do not
        # overwrite launcher-provided values for a managed multi-rank run.
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(self.world_size))
        local_rank = self.device.index if self.device.type == "cuda" and self.device.index is not None else 0
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{address}:{port}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )

    def _redis_client(self) -> object:
        try:
            import redis
        except ImportError as error:  # pragma: no cover - deployment dependency failure.
            raise RuntimeError("redis-py is required to run the Learner worker.") from error
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        url = redis_cfg.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("run.redis.url must be a non-empty string.")
        return redis.Redis.from_url(url, decode_responses=False)

    def _broadcast_command(self, command: object) -> object:
        if not _distributed_active():
            return command
        objects = [command if self.rank == 0 else None]
        dist.broadcast_object_list(objects, src=0, device=self.device)
        return objects[0]

    def run_forever(self, stop_event: object) -> None:
        """Consume bounded Learner turns on rank zero and coordinate all ranks."""

        if not callable(getattr(stop_event, "is_set", None)):
            raise TypeError("stop_event must provide is_set().")
        if not self._initialized:
            self.initialize_engine()
        client = self._redis_client() if self.rank == 0 else None
        # This is intentionally fixed.  It bounds shutdown responsiveness
        # without making the framework's asynchronous queue mechanics part of
        # each run configuration.
        timeout = 1.0
        if self.rank == 0:
            print(
                f"[Learner GPU {self.gpu_id}] Listening on {self.queue_keys()['learner_train']}; "
                f"max_batches_per_turn={self.max_batches_per_turn}.",
                flush=True,
            )

        while True:
            command: object = {"kind": "idle"}
            if self.rank == 0:
                if stop_event.is_set():
                    command = {"kind": "stop"}
                else:
                    try:
                        payload = client.brpop(self.queue_keys()["learner_train"], timeout=timeout)  # type: ignore[union-attr]
                    except Exception as error:
                        self._publish_event(client, f"Redis receive failed: {error}")
                        payload = None
                    if payload:
                        _key, raw = payload
                        try:
                            decoded = pickle.loads(raw)
                            if not isinstance(decoded, Mapping):
                                raise TypeError("Redis payload did not contain a signal mapping.")
                            batches: list[Signals] = [dict(decoded)]
                            for _ in range(self.max_batches_per_turn - 1):
                                try:
                                    extra = client.rpop(self.queue_keys()["learner_train"])  # type: ignore[union-attr]
                                except Exception:
                                    break
                                if extra is None:
                                    break
                                try:
                                    extra_decoded = pickle.loads(extra)
                                    if not isinstance(extra_decoded, Mapping):
                                        raise TypeError("Redis payload did not contain a signal mapping.")
                                    batches.append(dict(extra_decoded))
                                except Exception as error:
                                    self._publish_event(client, f"Could not decode Learner input: {error}")
                            command = {"kind": "batches", "signals": batches}
                        except Exception as error:
                            self._publish_event(client, f"Could not decode Learner input: {error}")
            command = self._broadcast_command(command)
            if not isinstance(command, Mapping):
                raise RuntimeError("Learner distributed command must be a mapping.")
            kind = command.get("kind")
            if kind == "stop":
                return
            if kind != "batches":
                continue
            raw_batches = command.get("signals")
            if not isinstance(raw_batches, list) or not raw_batches:
                raise RuntimeError("Learner batches command lacks signal mappings.")
            if not all(isinstance(batch, Mapping) for batch in raw_batches):
                raise RuntimeError("Learner batches command contains a non-mapping payload.")

            waiting_started = time.perf_counter()
            if not self.gpu_gate.acquire(stop_event):
                continue
            gate_wait_seconds = time.perf_counter() - waiting_started
            turn_started = time.perf_counter()
            completed_count = 0
            model_load_seconds = 0.0
            model_offload_seconds = 0.0
            completed: list[tuple[Signals, float]] = []
            try:
                model_load_seconds = self._activate_model()
                for batch in raw_batches:
                    batch_started = time.perf_counter()
                    result = self.process_training_batch(dict(batch))
                    completed.append((result, time.perf_counter() - batch_started))
                self._synchronize_cuda()
                model_offload_seconds = self._offload_model()
                if self.rank == 0:
                    pipeline = client.pipeline()  # type: ignore[union-attr]
                    for result, processing_seconds in completed:
                        timed = self._with_worker_timing(
                            result,
                            worker=self.node_id,
                            processing_seconds=processing_seconds,
                            gate_wait_seconds=gate_wait_seconds,
                            model_load_seconds=model_load_seconds,
                            model_offload_seconds=model_offload_seconds,
                        )
                        safe_result = _deep_cpu_detach(timed)
                        pipeline.lpush(
                            self.queue_keys()["controller_train_results"],
                            pickle.dumps(safe_result, protocol=pickle.HIGHEST_PROTOCOL),
                        )
                    pipeline.execute()
                completed_count = len(completed)
            finally:
                try:
                    if self.cpu_offload:
                        model_offload_seconds += self._offload_model()
                finally:
                    # Offload can fail; do not strand a shared-GPU peer.
                    gate_hold_seconds = time.perf_counter() - turn_started
                    self.gpu_gate.release()
                print(
                    f"[Learner {self.node_id} rank {self.rank} GPU {self.gpu_id}] turn batches="
                    f"{completed_count}/{len(raw_batches)}, gate_wait={gate_wait_seconds:.3f}s, "
                    f"gate_hold={gate_hold_seconds:.3f}s, model_load={model_load_seconds:.3f}s, "
                    f"model_offload={model_offload_seconds:.3f}s.",
                    flush=True,
                )
                self.gpu_gate.yield_after_release(stop_event, delay_sec=2.0)

    def _synchronize_cuda(self) -> None:
        """Finish this Learner's queued CUDA work before releasing its Gate."""

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _with_worker_timing(
        self,
        signals: Signals,
        *,
        worker: str,
        processing_seconds: float,
        gate_wait_seconds: float,
        model_load_seconds: float,
        model_offload_seconds: float,
    ) -> Signals:
        """Preserve upstream timing and append one Worker-local measurement."""

        result: Signals = dict(signals)
        raw_timing = result.get("timing", {})
        timing = dict(raw_timing) if isinstance(raw_timing, Mapping) else {}
        raw_workers = timing.get("workers", {})
        workers = dict(raw_workers) if isinstance(raw_workers, Mapping) else {}
        workers[worker] = {
            "processing_seconds": max(0.0, float(processing_seconds)),
            "gate_wait_seconds": max(0.0, float(gate_wait_seconds)),
            "cpu_offload": self.cpu_offload,
            "model_load_seconds": max(0.0, float(model_load_seconds)),
            "model_offload_seconds": max(0.0, float(model_offload_seconds)),
            "model_load_count": self._model_load_count,
            "model_offload_count": self._model_offload_count,
        }
        timing["workers"] = workers
        result["timing"] = timing
        # Keep Learner-owned execution timing alongside the other framework
        # metrics so Controller can log a complete Learner component from the
        # one Redis result without reconstructing a Worker-local view.
        raw_learner_metrics = result.get("learner_metrics")
        if isinstance(raw_learner_metrics, Mapping):
            learner_metrics = dict(raw_learner_metrics)
            learner_metrics.update(workers[worker])
            result["learner_metrics"] = learner_metrics
        return result

    def _publish_event(self, client: object | None, message: str) -> None:
        if client is None:
            return
        event = {
            "source": "learner",
            "rank": self.rank,
            "message": str(message),
            "timestamp": time.time(),
        }
        try:
            client.lpush(
                self.queue_keys()["controller_events"],
                pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL),
            )
        except Exception as error:
            print(f"[Learner] Could not publish event: {error}", flush=True)

    def close(self) -> None:
        """Release model references on a worker retry or controlled shutdown."""

        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._model_on_gpu = False
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


PluginFactory = Callable[[object], AlgorithmPlugin]


def _build_plugin(
    factory: PluginFactory | None,
    tokenizer: object,
) -> AlgorithmPlugin | None:
    if factory is None:
        return None
    plugin = factory(tokenizer)
    if not isinstance(plugin, AlgorithmPlugin):
        raise TypeError("plugin_factory must return an AlgorithmPlugin.")
    return plugin


class LearnerManager:
    """Launch one collective DeepSpeed Learner group over ``learner.gpu_ids``.

    Unlike Actor's independent vLLM processes, all Learner ranks belong to one
    distributed optimizer.  A failed rank therefore restarts the *whole*
    group; restarting an individual NCCL rank would leave surviving ranks
    stuck in stale collectives.  The exception is a group assigned to a
    shared GPU gate: a hard-killed rank may have leaked that gate's semaphore,
    so the Manager records a terminal failure for Controller instead of
    restarting into a permanent wait.
    """

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
        learner_cfg = _mapping(self.config.get("learner"), "learner")
        run_cfg = _mapping(self.config.get("run"), "run")
        runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
        raw_gpu_ids = learner_cfg.get("gpu_ids", [])
        if not isinstance(raw_gpu_ids, list):
            raise ValueError("learner.gpu_ids must be a list of GPU ids.")
        self.gpu_ids = [self._validated_gpu_id(item) for item in raw_gpu_ids]
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("learner.gpu_ids may not contain duplicates.")
        if not self.gpu_ids:
            raise ValueError("learner.gpu_ids must contain at least one GPU id.")
        minimum = float(runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
        if minimum < 3:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb may not be below 3 GB.")
        self._context = mp.get_context(str(runtime_cfg.get("process_start_method", "spawn")))
        self.stop_event = self._context.Event()
        self.processes: dict[int, object] = {}
        # Controller owns final cleanup.  The supervisor only records this
        # state and requests cooperative shutdown, avoiding a restart after a
        # hard exit may have leaked a shared-GPU semaphore permit.
        self.fatal_error: str | None = None
        # Each rank signals after its distributed engine, model, and plugin
        # are initialized.  Controller waits before loading Teacher/Actor on
        # a shared local GPU.
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
        self._start_group()
        print(f"[Learner Manager] Started {len(self.processes)} Learner rank(s).", flush=True)

    def _start_group(self) -> None:
        if self.processes:
            raise RuntimeError("Learner process group is already running.")
        for rank, gpu_id in enumerate(self.gpu_ids):
            ready_event = self._context.Event()
            self.ready_events[rank] = ready_event
            process = self._context.Process(
                target=_learner_worker_main,
                args=(
                    rank,
                    len(self.gpu_ids),
                    gpu_id,
                    self.config,
                    self.plugin_factory,
                    self.stop_event,
                    ready_event,
                    self.gpu_gates.for_gpu(gpu_id) if self.gpu_gates is not None else None,
                ),
                name=f"FrugalOPL-Learner-rank{rank}-GPU{gpu_id}",
            )
            process.start()
            self.processes[rank] = process

    def _shared_gpu_ids(self) -> tuple[int, ...]:
        """Return physical GPUs whose Learner ranks share another Worker."""

        if self.gpu_gates is None:
            return ()
        return tuple(gpu_id for gpu_id in self.gpu_ids if self.gpu_gates.for_gpu(gpu_id).is_shared)

    def _record_terminal_failure(self, message: str) -> None:
        """Make a failed group visible to Controller and stop sibling ranks."""

        if self.fatal_error is None:
            self.fatal_error = message
            print(f"[Learner Manager FATAL] {message}", flush=True)
        self.stop_event.set()

    def wait_until_ready(self, *, timeout_sec: float = 300.0) -> None:
        """Block until every initially launched Learner rank is usable."""

        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + float(timeout_sec)
        pending = set(self.processes)
        while pending:
            for rank in tuple(pending):
                event = self.ready_events.get(rank)
                is_set = getattr(event, "is_set", None)
                if callable(is_set) and is_set():
                    pending.remove(rank)
                    continue
                process = self.processes.get(rank)
                alive = getattr(process, "is_alive", None)
                if not callable(alive) or not alive():
                    exitcode = getattr(process, "exitcode", None)
                    raise RuntimeError(
                        f"Learner rank {rank} exited before becoming ready (exitcode={exitcode})."
                    )
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waiting = ", ".join(str(rank) for rank in sorted(pending))
                raise TimeoutError(f"Timed out waiting for Learner rank(s) {waiting} to become ready.")
            self.stop_event.wait(min(0.1, remaining))

    def join(self) -> None:
        run_cfg = _mapping(self.config.get("run"), "run")
        restart_cfg = _mapping(
            _mapping(run_cfg.get("runtime", {}), "run.runtime").get("restart", {}),
            "run.runtime.restart",
        )
        restart_enabled = bool(restart_cfg.get("enabled", True))
        delay = max(0.0, float(restart_cfg.get("worker_restart_delay_sec", 5.0)))
        poll = max(0.05, float(restart_cfg.get("manager_poll_interval_sec", 2.0)))
        try:
            while not self.stop_event.is_set() and self.processes:
                failed = [process for process in self.processes.values() if not process.is_alive()]
                if not failed:
                    self.stop_event.wait(poll)
                    continue
                exit_codes = [process.exitcode for process in failed]
                for process in failed:
                    process.join(timeout=0.1)
                shared_gpu_ids = self._shared_gpu_ids()
                if shared_gpu_ids:
                    rendered_gpu_ids = ", ".join(str(gpu_id) for gpu_id in shared_gpu_ids)
                    self._record_terminal_failure(
                        f"Learner group exited ({exit_codes}) while using shared GPU gate(s) {rendered_gpu_ids}; "
                        "automatic restart is unsafe because a hard exit can leave a semaphore locked."
                    )
                    return
                if not restart_enabled:
                    self._record_terminal_failure(
                        f"Learner group exited ({exit_codes}) and worker restart is disabled."
                    )
                    return
                if self.stop_event.is_set():
                    return
                self._stop_group(force=True)
                print(
                    f"[Learner Manager] Learner rank failed ({exit_codes}); restarting the whole group.",
                    flush=True,
                )
                if self.stop_event.wait(delay):
                    return
                self._start_group()
        except KeyboardInterrupt:
            self.stop()

    def request_stop(self) -> None:
        """Broadcast cooperative cancellation without joining the rank group."""

        self.stop_event.set()

    def _shutdown_grace_sec(self) -> float:
        """Return the normal cleanup window for this collective group."""

        if self._shared_gpu_ids():
            return _SHARED_GPU_SHUTDOWN_GRACE_SEC
        return _EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC

    @staticmethod
    def _join_processes(processes: tuple[object, ...], *, timeout_sec: float) -> None:
        """Join ranks within a single total deadline, not once per rank."""

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _alive_processes(processes: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(process for process in processes if process.is_alive())

    @staticmethod
    def _signal_processes(processes: tuple[object, ...], method_name: str) -> None:
        for process in processes:
            method = getattr(process, method_name, None)
            if method_name == "kill" and not callable(method):
                # WSL's multiprocessing Process exposes ``kill``.  Preserve a
                # terminate fallback for alternate adapters instead of
                # silently dropping an unreaped child.
                method = getattr(process, "terminate", None)
            if callable(method):
                method()

    def _stop_group(self, *, force: bool) -> None:
        """Stop only ranks owned by this Manager and reap every survivor.

        ``force=True`` is the pre-existing failed-group recovery path.  It
        keeps its immediate escalation but now follows SIGTERM with SIGKILL if
        the Worker signal handler cannot regain control.  ``force=False`` is
        the normal Controller path and first gives shared-GPU teardown a
        longer, explicit cooperative window.
        """

        processes = tuple(self.processes.values())
        if not processes:
            return

        if force:
            remaining = self._alive_processes(processes)
            if remaining:
                self._signal_processes(remaining, "terminate")
            self._join_processes(processes, timeout_sec=_TERMINATE_SHUTDOWN_GRACE_SEC)
        else:
            grace_sec = self._shutdown_grace_sec()
            self._join_processes(processes, timeout_sec=grace_sec)
            remaining = self._alive_processes(processes)
            if remaining:
                print(
                    f"[Learner Manager] Cooperative shutdown exceeded {grace_sec:.0f}s; "
                    f"sending terminate to {len(remaining)} Learner rank(s).",
                    flush=True,
                )
                self._signal_processes(remaining, "terminate")
                self._join_processes(remaining, timeout_sec=_TERMINATE_SHUTDOWN_GRACE_SEC)

        remaining = self._alive_processes(processes)
        if remaining:
            # SIGTERM is intentionally handled as a cooperative event by the
            # rank entrypoint.  A child stuck in native code may not run that
            # handler, so explicitly reap it rather than leaving a detached
            # NCCL/CUDA process behind this Manager.
            print(
                f"[Learner Manager] Terminate grace exceeded; killing {len(remaining)} Learner rank(s).",
                flush=True,
            )
            self._signal_processes(remaining, "kill")
            self._join_processes(remaining, timeout_sec=_KILL_SHUTDOWN_JOIN_SEC)

        remaining = self._alive_processes(processes)
        if remaining:
            raise RuntimeError(
                "Learner Manager could not reap all owned ranks after forced shutdown; "
                f"{len(remaining)} process(es) remain alive."
            )
        self.processes.clear()

    def finalize_stop(self) -> None:
        """Finish a previously requested cooperative shutdown."""

        self._stop_group(force=False)

    def stop(self) -> None:
        """Standalone Manager shutdown, retained for direct Worker CLIs."""

        self.request_stop()
        self.finalize_stop()


def _learner_worker_main(
    rank: int,
    world_size: int,
    gpu_id: int,
    config: Mapping[str, object],
    plugin_factory: PluginFactory | None,
    stop_event: object,
    ready_event: object,
    gpu_gate: GpuGate | None,
) -> None:
    # GPU isolation must occur before any CUDA operation in this child.  Torch
    # import itself is harmless; no CUDA context is created until Learner
    # initialization below.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["FRUGALOPL_PHYSICAL_GPU_ID"] = str(gpu_id)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)

    def request_stop(_signum: int, _frame: object) -> None:
        setter = getattr(stop_event, "set", None)
        if callable(setter):
            setter()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run_cfg = _mapping(config.get("run"), "run")
    runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
    restart_cfg = _mapping(runtime_cfg.get("restart", {}), "run.runtime.restart")
    retry_delay = max(0.0, float(restart_cfg.get("worker_restart_delay_sec", 5.0)))
    # Controller's startup ordering only covers the first launch.  A later
    # Learner retry can otherwise create a CUDA context, initialize NCCL, or
    # migrate its model while an Actor/Teacher already computes on this card.
    # Keep the registry gate through all of that setup.  Direct construction
    # keeps a no-op gate for standalone use and CPU contract tests.
    initialization_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=gpu_id)

    def cleanup_resources(*, gate_is_held: bool) -> None:
        """Tear down this rank without racing a shared-card peer.

        Both model cleanup and NCCL process-group destruction can issue CUDA
        work.  The latter is especially easy to miss because it normally
        runs only on retries or shutdown.  Keep both operations under the
        same physical-GPU permit used by initialization and training turns.
        ``gate_is_held`` avoids a recursive acquire on an initialization
        failure, where the caller already owns the permit.
        """

        nonlocal learner
        if learner is None and not dist.is_initialized():
            return
        acquired_here = False
        if not gate_is_held:
            # Deliberately do not pass stop_event: controlled shutdown still
            # needs to relinquish CUDA/NCCL resources before this process
            # exits, even after that event has been set.
            initialization_gate.acquire()
            acquired_here = True
        try:
            if learner is not None:
                learner.close()
                learner = None
            if dist.is_initialized():
                print(f"[Learner rank {rank}] Destroying distributed process group.", flush=True)
                try:
                    dist.destroy_process_group()
                except Exception as error:
                    # Cleanup must continue so the GPU gate is released even
                    # if NCCL teardown itself reports a late error.  Do not
                    # swallow it silently: a smoke log needs to distinguish
                    # an uncalled destroy from a failed destroy.
                    print(
                        f"[Learner rank {rank}] WARNING: destroy_process_group failed: {error}",
                        flush=True,
                    )
                else:
                    print(f"[Learner rank {rank}] Destroyed distributed process group.", flush=True)
        finally:
            if acquired_here:
                initialization_gate.release()

    while not stop_event.is_set():
        learner: Learner | None = None
        try:
            if not initialization_gate.acquire(stop_event):
                # Stop was requested while another Worker owned this shared
                # card.  No per-attempt model was constructed yet.
                return
            try:
                # Construction can query CUDA; distributed initialization,
                # DeepSpeed setup, and model migration certainly use it.
                learner = Learner(
                    config,
                    rank=rank,
                    world_size=world_size,
                    gpu_id=gpu_id,
                    gpu_gate=initialization_gate,
                )
                learner.initialize_distributed()
                learner.initialize_engine()
                tokenizer = learner._require_tokenizer()
                plugin = _build_plugin(plugin_factory, tokenizer)
                if plugin is not None:
                    learner.set_plugin(plugin)
            except BaseException:
                # Model/optimizer teardown can call CUDA APIs.  Do it before
                # releasing the permit so a failed retry cannot overlap a
                # peer's normal GPU turn.
                cleanup_resources(gate_is_held=True)
                raise
            finally:
                initialization_gate.release()
            mark_ready = getattr(ready_event, "set", None)
            if callable(mark_ready):
                mark_ready()
            learner.run_forever(stop_event)
            return
        except Exception as error:
            print(
                f"[Learner rank {rank} FATAL] {error}\n{traceback.format_exc()}",
                flush=True,
            )
            cleanup_resources(gate_is_held=False)
            if world_size > 1:
                # The Manager restarts the whole group.  Avoid attempting an
                # in-place single-rank recovery against a stale collective.
                # A lone-rank retry cannot repair a collective group.  Exit
                # so LearnerManager can stop and restart every rank together.
                raise
            waiter = getattr(stop_event, "wait", None)
            if callable(waiter):
                waiter(retry_delay)
            else:
                time.sleep(retry_delay)
        finally:
            cleanup_resources(gate_is_held=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FrugalOPL's generic DeepSpeed Learner.")
    parser.add_argument("--framework-config", required=True, help="Path to a framework profile YAML.")
    parser.add_argument("--algorithm-config", default=None, help="Optional algorithm overlay YAML file.")
    args = parser.parse_args()
    # Import lazily to keep this module self-contained for CPU unit tests.
    from engine.actor import load_merged_config

    config = load_merged_config(args.framework_config, args.algorithm_config)
    factory = plugin_factory_from_config(config)
    manager = LearnerManager(config, plugin_factory=factory)
    manager.start()
    manager.join()


if __name__ == "__main__":
    main()


__all__ = [
    "ComputeMode",
    "DEFAULT_LOGIT_SIGNALS",
    "Learner",
    "LearnerManager",
    "PluginFactory",
]
