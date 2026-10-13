"""The no-gradient Teacher worker used by OPD-style algorithm plugins.

Teacher is deliberately smaller than :mod:`engine.learner`: it owns one
read-only Transformers model, consumes Actor trajectories, lets the selected
plugin derive detached Teacher signals, and forwards the enriched dictionary
to Learner.  It has no optimizer, LoRA, reference-policy mode, DeepSpeed, or
gradient-bearing API.

When Teacher shares a physical GPU with another Worker, it cooperatively
acquires that GPU's execution gate for at most
``teacher.forward.max_batches_per_turn`` logical batches.  It completes and
publishes that bounded turn before releasing the permit, while Controller's
bounded in-flight batches continue to limit how far Actor may run ahead.
Workers on distinct cards can still pipeline through Redis; only Workers
sharing a physical card are serialized.

The algorithm-facing surface remains only ``self.compute_logits(...)`` inside
the Teacher node's configured plugin hook. It returns CPU-detached, ragged
token signals in the exact rollout order supplied by Actor.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch

from algorithm_plugins import plugin_factory_from_config
from algorithm_plugins.algorithm_base import AlgorithmPlugin, Signals, _teacher_plugin_scope
from engine.gpu_gate import GpuGate, GpuGateRegistry
from engine.pipeline import PipelineNode, resolve_training_pipeline, resolve_worker_config
from engine.learner import (
    _as_ragged_token_list,
    _deep_cpu_detach,
    _mapping,
    _normalize_logit_signals,
    _run_name_and_redis,
    _torch_dtype,
)


# Teacher model teardown may release CUDA allocations while it owns a shared
# physical-GPU gate.  This grace is a lifecycle safety boundary, not an
# experiment option: it prevents a normal Controller shutdown from killing a
# permit holder before Learner/Actor have had a chance to clean up.
_EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC = 5.0
_SHARED_GPU_SHUTDOWN_GRACE_SEC = 30.0
_TERMINATE_SHUTDOWN_GRACE_SEC = 5.0
_KILL_SHUTDOWN_JOIN_SEC = 5.0


def _gpu_memory_info(gpu_id: int) -> tuple[int, int]:
    """Read device-wide free/total VRAM without creating a CUDA context."""

    command = [
        "nvidia-smi",
        "--id",
        str(gpu_id),
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"Unexpected nvidia-smi response: {completed.stdout!r}")
    free_mib, total_mib = (int(field) for field in fields)
    return free_mib * 1024**2, total_mib * 1024**2


def _minimum_free_bytes(runtime_cfg: Mapping[str, object]) -> int:
    """Return the configured device-wide free-memory reserve.

    Local configurations keep their 3 GiB safety reserve.  A managed server
    run may explicitly set this to zero so allocation failures, rather than a
    fabricated local-desktop threshold, determine whether it OOMs.
    """

    minimum_gb = float(runtime_cfg.get("minimum_free_gpu_memory_gb", 3))
    if minimum_gb < 0:
        raise ValueError("run.runtime.minimum_free_gpu_memory_gb must be non-negative.")
    minimum = int(minimum_gb * 1024**3)
    return minimum


class Teacher:
    """One read-only Teacher process bound to one GPU (or CPU in tests)."""

    def __init__(
        self,
        config: Mapping[str, object],
        plugin: AlgorithmPlugin | None = None,
        *,
        node_id: str = "teacher",
        gpu_id: int | None = None,
        device: torch.device | str | None = None,
        gpu_gate: GpuGate | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        self.config: dict[str, object] = copy.deepcopy(dict(config))
        self.run_cfg = _mapping(self.config.get("run"), "run")
        self.policy_cfg = _mapping(self.run_cfg.get("policy"), "run.policy")
        self.node_id = node_id
        self.pipeline = resolve_training_pipeline(self.config)
        self.pipeline_node: PipelineNode = self.pipeline.node(node_id)
        if self.pipeline_node.worker_class != "Teacher":
            raise ValueError(f"Pipeline node {node_id!r} is not a Teacher.")
        self.teacher_cfg = resolve_worker_config(self.config, node_id)
        self.forward_cfg = _mapping(self.teacher_cfg.get("forward", {}), f"{node_id}.forward")
        self.runtime_cfg = _mapping(self.run_cfg.get("runtime", {}), "run.runtime")
        self.cpu_offload = self.pipeline_node.cpu_offload

        configured_gpu_ids = self.teacher_cfg.get("gpu_ids", [])
        if not isinstance(configured_gpu_ids, list):
            raise ValueError(f"{self.node_id}.gpu_ids must be a list of GPU ids.")
        if gpu_id is None:
            gpu_id = int(configured_gpu_ids[0]) if configured_gpu_ids else 0
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer.")
        if configured_gpu_ids and gpu_id not in configured_gpu_ids:
            raise ValueError(f"gpu_id must be one of {self.node_id}.gpu_ids.")
        self.gpu_id = gpu_id
        if gpu_gate is not None and not isinstance(gpu_gate, GpuGate):
            raise TypeError("gpu_gate must be a GpuGate when provided.")
        if gpu_gate is not None and gpu_gate.physical_gpu_id != self.gpu_id:
            raise ValueError("gpu_gate.physical_gpu_id must match Teacher's physical gpu_id.")
        # Direct construction in CPU/unit tests has no Controller registry.
        # An exclusive no-op gate retains the same call path without requiring
        # synchronization infrastructure.
        self.gpu_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=self.gpu_id)
        self.device = torch.device(device) if device is not None else self._default_device()
        self.plugin = plugin
        self.tokenizer: object | None = None
        self.model: torch.nn.Module | None = None
        self._initialized = False
        self._model_on_device = False
        self._model_load_seconds = 0.0
        self._model_offload_seconds = 0.0
        self._model_load_count = 0
        self._model_offload_count = 0

        if "micro_batch_size" in self.forward_cfg:
            raise ValueError(
                "teacher.forward.micro_batch_size has been removed; Teacher always forwards one rollout at a time."
            )
        self.forward_chunk_size = int(self.forward_cfg.get("chunk_size", 512))
        self.max_batches_per_turn = int(self.forward_cfg.get("max_batches_per_turn", 1))
        if self.forward_chunk_size < 1:
            raise ValueError("teacher.forward.chunk_size must be positive.")
        if self.max_batches_per_turn < 1:
            raise ValueError("teacher.forward.max_batches_per_turn must be positive.")
        self.max_sequence_length = int(self.teacher_cfg.get("max_sequence_length", 0))
        if self.max_sequence_length < 1:
            raise ValueError(f"{node_id}.max_sequence_length must be positive.")

    def _default_device(self) -> torch.device:
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    def set_plugin(self, plugin: AlgorithmPlugin) -> None:
        if not isinstance(plugin, AlgorithmPlugin):
            raise TypeError("plugin must inherit AlgorithmPlugin.")
        self.plugin = plugin

    def queue_keys(self) -> dict[str, str]:
        name, prefix = _run_name_and_redis(self.config)
        root = f"{prefix}:{name}"
        downstream = self.pipeline.successor_of(self.node_id)
        return {
            "teacher_train": f"{root}:{self.node_id}:train",
            "learner_train": f"{root}:{downstream}:train",
            "controller_events": f"{root}:controller:events",
        }

    def initialize_engine(self) -> None:
        """Load the fixed Teacher model in bfloat16 / configured dtype."""

        if self._initialized:
            return
        self._validate_memory_before_load()
        configured_path = self.teacher_cfg.get("model_name_or_path") or self.policy_cfg.get("model_name_or_path")
        model_path = os.path.expanduser(str(configured_path or "")).strip()
        if not model_path:
            raise ValueError("teacher.model_name_or_path or run.policy.model_name_or_path must be configured.")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - deployment dependency failure.
            raise RuntimeError("transformers is required to initialize the Teacher.") from error

        dtype = _torch_dtype(self.teacher_cfg.get("dtype", self.policy_cfg.get("dtype", "bfloat16")))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._validate_tokenizer_compatibility(AutoTokenizer, model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.config.use_cache = False
        self._force_mask_compatible_attention(self.model)
        self._validate_model_load_headroom(self.model)
        self.model.to(self.device)
        self.model.eval()
        self._model_on_device = True
        self._validate_memory_after_load()
        if self.cpu_offload:
            self._offload_model()
        self._initialized = True

    def _activate_model(self) -> float:
        if not self.cpu_offload or self.device.type != "cuda" or not torch.cuda.is_available() or self._model_on_device:
            return 0.0
        model = self._require_model()
        self._validate_model_load_headroom(model)
        started = time.perf_counter()
        model.to(self.device)
        self._synchronize_device()
        self._model_on_device = True
        self._validate_memory_after_load()
        elapsed = time.perf_counter() - started
        self._model_load_seconds += elapsed
        self._model_load_count += 1
        return elapsed

    def _offload_model(self) -> float:
        if not self.cpu_offload or self.device.type != "cuda" or not torch.cuda.is_available() or not self._model_on_device:
            return 0.0
        self._synchronize_device()
        model = self._require_model()
        started = time.perf_counter()
        model.to("cpu")
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_on_device = False
        elapsed = time.perf_counter() - started
        self._model_offload_seconds += elapsed
        self._model_offload_count += 1
        return elapsed

    def _validate_model_load_headroom(self, model: torch.nn.Module) -> None:
        """Keep the configured device-wide reserve after a model moves in."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        free, _total = _gpu_memory_info(self.gpu_id)
        minimum = _minimum_free_bytes(self.runtime_cfg)
        model_bytes = sum(item.numel() * item.element_size() for item in (*model.parameters(), *model.buffers()))
        required = minimum + model_bytes
        if free < required:
            raise MemoryError(
                f"Teacher {self.node_id} model load needs {required / 1024**3:.2f} GB "
                f"free, but GPU has {free / 1024**3:.2f} GB."
            )

    def _validate_tokenizer_compatibility(self, tokenizer_factory: object, teacher_path: str) -> None:
        """Guard the raw-token Teacher path used by ordinary PG-OPD.

        PG-OPD scores Actor's exact Student token IDs.  A matching model
        family name alone is insufficient: an ID mismatch would silently turn
        the signal into unrelated vocabulary entries.  Algorithms that pass
        their own ``prompt_ids`` still share response IDs, so they need the
        same condition.  This is a non-configurable input contract: an
        algorithm with an explicit token mapping must apply it before asking
        Teacher to score those IDs.
        """

        student_path = os.path.expanduser(str(self.policy_cfg.get("model_name_or_path", ""))).strip()
        if not student_path or student_path == teacher_path:
            return
        factory = getattr(tokenizer_factory, "from_pretrained", None)
        if not callable(factory):  # pragma: no cover - AutoTokenizer invariant.
            raise RuntimeError("Teacher tokenizer factory does not provide from_pretrained().")
        student_tokenizer = factory(student_path, trust_remote_code=True)
        teacher_vocab = getattr(self.tokenizer, "get_vocab", None)
        student_vocab = getattr(student_tokenizer, "get_vocab", None)
        if not callable(teacher_vocab) or not callable(student_vocab):
            raise TypeError("Teacher and policy tokenizers must provide get_vocab() for ID compatibility checks.")
        if teacher_vocab() != student_vocab():
            raise ValueError(
                "Teacher and policy tokenizer token-to-ID mappings differ. PG-OPD cannot score Actor token IDs "
                "without an explicit mapping."
            )

    @staticmethod
    def _force_mask_compatible_attention(model: torch.nn.Module) -> None:
        setter = getattr(model, "set_attn_implementation", None)
        if callable(setter):
            setter("sdpa")
            return
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            setattr(config, "_attn_implementation", "sdpa")

    def _validate_memory_before_load(self) -> None:
        if self.device.type != "cuda":
            return
        free, _total = _gpu_memory_info(self.gpu_id)
        minimum = _minimum_free_bytes(self.runtime_cfg)
        if free < minimum:
            raise MemoryError(
                f"Refusing to initialize Teacher: GPU has only {free / 1024**3:.2f} GB free; "
                f"the configured free-memory reserve is {minimum / 1024**3:.2f} GB."
            )

    def _validate_memory_after_load(self) -> None:
        if self.device.type != "cuda":
            return
        free, _total = _gpu_memory_info(self.gpu_id)
        minimum = _minimum_free_bytes(self.runtime_cfg)
        if free < minimum:
            raise MemoryError(
                f"Teacher initialization left only {free / 1024**3:.2f} GB free; "
                f"the configured free-memory reserve is {minimum / 1024**3:.2f} GB."
            )

    def _require_model(self) -> torch.nn.Module:
        if not isinstance(self.model, torch.nn.Module):
            raise RuntimeError("Teacher model is not initialized.")
        return self.model

    def _require_tokenizer(self) -> object:
        if self.tokenizer is None:
            raise RuntimeError("Teacher tokenizer is not initialized.")
        return self.tokenizer

    def _plugin_compute_logits(
        self,
        batch: Signals,
        *,
        mode: str | None,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
        return_signals: Mapping[str, object] | None,
    ) -> Signals:
        """Return detached Teacher scores for every rollout in input order."""

        if mode is not None:
            raise ValueError("Teacher compute_logits() has no mode.")
        if prompts is not None and prompt_ids is not None:
            raise ValueError("Use either prompts or prompt_ids, not both.")
        responses, default_prompts = self._trajectory_token_lists(batch)
        if prompts is not None:
            prompt_rows = self._encode_prompts(prompts)
        elif prompt_ids is not None:
            prompt_rows = _as_ragged_token_list(prompt_ids, "prompt_ids")
        else:
            prompt_rows = default_prompts
        if len(prompt_rows) != len(responses):
            raise ValueError("Teacher prompt IDs and response IDs must have the same rollout length.")
        if not responses:
            raise ValueError("Teacher compute_logits() received no rollouts.")
        if any(row.numel() == 0 for row in prompt_rows):
            raise ValueError("Every Teacher prompt must contain at least one token.")
        if any(row.numel() == 0 for row in responses):
            raise ValueError("Every Teacher response must contain at least one token.")

        required = _normalize_logit_signals(return_signals)
        collected: dict[str, list[torch.Tensor]] = {}
        for prompt, response in zip(prompt_rows, responses, strict=True):
            prepared = self._prepare_forward_rollout(prompt, response)
            rollout_signals = self._compute_token_signals(prepared, required)
            for key, signal in rollout_signals.items():
                if not isinstance(signal, torch.Tensor):  # pragma: no cover - local invariant.
                    raise RuntimeError(f"Teacher produced a non-tensor signal at {key!r}.")
                if int(signal.size(0)) != 1:  # pragma: no cover - local invariant.
                    raise RuntimeError(f"Teacher produced a non-singleton rollout signal at {key!r}.")
                collected.setdefault(key, []).append(signal.squeeze(0).detach().cpu().contiguous())
        return collected

    def _trajectory_token_lists(self, batch: Mapping[str, object]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        trajectory = batch.get("traj")
        source = trajectory if isinstance(trajectory, Mapping) else batch
        if not isinstance(source, Mapping):  # pragma: no cover - guarded above.
            raise TypeError("Teacher batch must contain a trajectory mapping.")
        if "response_ids" not in source or "prompt_ids" not in source:
            raise ValueError("Teacher compute_logits() requires traj.prompt_ids and traj.response_ids.")
        responses = _as_ragged_token_list(source["response_ids"], "traj.response_ids")
        prompts = _as_ragged_token_list(source["prompt_ids"], "traj.prompt_ids")
        if len(prompts) != len(responses):
            raise ValueError("traj.prompt_ids and traj.response_ids must have the same rollout length.")
        return responses, prompts

    def _encode_prompts(self, prompts: Sequence[str]) -> list[torch.Tensor]:
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
            raise TypeError("prompts must be a sequence of final rendered prompt strings.")
        tokenizer = self._require_tokenizer()
        tokenizer_call = getattr(tokenizer, "__call__", None)
        if not callable(tokenizer_call):
            raise TypeError("Teacher tokenizer must be callable to encode prompt overrides.")
        output: list[torch.Tensor] = []
        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise TypeError("prompts must contain only strings.")
            encoded = tokenizer_call(prompt, add_special_tokens=False)
            if not isinstance(encoded, Mapping):
                raise TypeError("Teacher tokenizer prompt output must be a mapping.")
            raw_ids = encoded.get("input_ids")
            if isinstance(raw_ids, torch.Tensor):
                ids = raw_ids.detach().cpu().reshape(-1).tolist()
            elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
                ids = list(raw_ids[0]) if raw_ids and isinstance(raw_ids[0], Sequence) else list(raw_ids)
            else:
                raise TypeError(f"Teacher prompt tokenizer output for row {index} lacks token IDs.")
            if not ids:
                raise ValueError("Prompt overrides may not tokenize to an empty sequence.")
            output.append(torch.tensor([int(token_id) for token_id in ids], dtype=torch.long))
        return output

    def _prepare_forward_rollout(
        self,
        prompt: torch.Tensor,
        response: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Prepare one complete rollout without cross-rollout padding."""

        prompt_ids = prompt.to(device=self.device, dtype=torch.long).unsqueeze(0)
        response_ids = response.to(device=self.device, dtype=torch.long).unsqueeze(0)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        sequence_length = int(input_ids.size(1))
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "Teacher compute_logits() sequence length exceeds teacher.max_sequence_length "
                f"({sequence_length} > {self.max_sequence_length})."
            )
        model_limit = getattr(getattr(self._require_model(), "config", None), "max_position_embeddings", None)
        if model_limit is not None and sequence_length > int(model_limit):
            raise ValueError(
                "Teacher compute_logits() sequence length exceeds the model position limit "
                f"({sequence_length} > {int(model_limit)})."
            )
        position_ids = torch.arange(sequence_length, device=self.device, dtype=torch.long).unsqueeze(0)
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _compute_token_signals(
        self,
        prepared: Mapping[str, torch.Tensor],
        required: Mapping[str, object],
    ) -> Signals:
        module = self._require_model()
        prior_training = bool(module.training)
        try:
            module.eval()
            with torch.no_grad():
                hidden = self._forward_hidden(
                    module,
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    position_ids=prepared["position_ids"],
                )
                prompt_width = int(prepared["prompt_ids"].size(1))
                response_width = int(prepared["response_ids"].size(1))
                response_hidden = hidden[:, prompt_width - 1 : prompt_width - 1 + response_width, :]
                return self._signals_from_response_hidden(
                    module,
                    response_hidden=response_hidden,
                    response_ids=prepared["response_ids"],
                    required=required,
                )
        finally:
            if prior_training:
                module.train()
            else:
                module.eval()

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
        raise RuntimeError("Teacher model must expose a decoder backbone at .model.")

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
        raise RuntimeError("Teacher model must expose an lm_head.")

    def _forward_hidden(
        self,
        module: torch.nn.Module,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self._backbone(module)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Teacher backbone did not return last_hidden_state.")
        return hidden

    def _signals_from_response_hidden(
        self,
        module: torch.nn.Module,
        *,
        response_hidden: torch.Tensor,
        response_ids: torch.Tensor,
        required: Mapping[str, object],
    ) -> Signals:
        """Select sampled-token logprobs and optional top-k without full-L logits."""

        want_logprobs = bool(required.get("response_logprobs", True))
        want_entropy = bool(required.get("response_entropy", False))
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
                log_probs = float_logits - normalization.unsqueeze(-1)
                probabilities = torch.exp(log_probs)
                entropy = -(probabilities * log_probs).sum(dim=-1)
                outputs["response_entropy"].append(entropy)
            if want_topk:
                values, token_ids = torch.topk(float_logits - normalization.unsqueeze(-1), k=top_k, dim=-1)
                outputs["topk_token_ids"].append(token_ids.to(dtype=torch.int32))
                outputs["topk_logprobs"].append(values)
        return {key: torch.cat(chunks, dim=1) for key, chunks in outputs.items() if chunks}

    def process_training_batch(self, signals: Mapping[str, object]) -> Signals:
        """Run the Teacher plugin hook and preserve every upstream signal.

        Besides algorithm signals, a plugin may return one reserved,
        Worker-local telemetry envelope::

            {"teacher_metrics": {"algorithm_metric_name": detached_scalar}}

        The envelope is intentionally algorithm-agnostic. Teacher validates
        and converts its values to finite Python scalars. Dynamic Teacher
        nodes namespace it by their node ID before forwarding it to Learner.
        """

        if self.plugin is None:
            raise RuntimeError("Teacher training requires an AlgorithmPlugin.")
        source: Signals = copy.deepcopy(dict(signals))
        hook_name = self.pipeline_node.plugin_hook
        hook = getattr(self.plugin, hook_name, None)
        if not callable(hook):
            raise RuntimeError(f"Teacher plugin hook is unavailable: {hook_name!r}.")
        with _teacher_plugin_scope(self):
            additions = hook(copy.deepcopy(source))
        if not isinstance(additions, Mapping):
            raise TypeError(f"Teacher plugin hook {hook_name!r} must return a mapping of new signals.")
        if not all(isinstance(key, str) for key in additions):
            raise TypeError(f"Teacher plugin hook {hook_name!r} must return string signal names.")
        normalized_additions: Signals = dict(additions)
        if "teacher_metrics" in normalized_additions:
            metrics_key = f"{self.node_id}_metrics"
            normalized_additions[metrics_key] = self._normalize_plugin_metrics(
                normalized_additions.pop("teacher_metrics")
            )
        collisions = sorted(key for key in normalized_additions if key in source)
        if collisions:
            raise ValueError(
                f"Teacher plugin hook {hook_name!r} may only return new fields; it cannot replace "
                f"an upstream signal: {collisions}."
            )
        source.update(normalized_additions)
        return _deep_cpu_detach(source)  # type: ignore[return-value]

    @staticmethod
    def _normalize_plugin_metrics(value: object) -> dict[str, float | int]:
        """Validate a plugin's optional detached ``teacher_metrics`` envelope.

        Worker metrics cross process and Redis boundaries, so accepting only
        finite scalar values avoids accidentally serializing model tensors or
        algorithm state as telemetry.  Tensor scalars are supported for
        convenience but must already be detached; they are converted to Python
        numbers here so downstream Workers never mistake them for model data.
        """

        if not isinstance(value, Mapping):
            raise TypeError("teacher_metrics returned by a Teacher plugin_hook must be a mapping.")
        metrics: dict[str, float | int] = {}
        for raw_name, raw_value in value.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise TypeError("teacher_metrics keys must be non-empty strings.")
            name = raw_name.strip()
            if name in metrics:
                raise ValueError(f"teacher_metrics contains duplicate metric name after normalization: {name!r}.")

            scalar: object = raw_value
            if isinstance(scalar, torch.Tensor):
                if scalar.numel() != 1:
                    raise ValueError(f"teacher_metrics[{name!r}] must be a scalar tensor.")
                if scalar.requires_grad:
                    raise ValueError(f"teacher_metrics[{name!r}] must be detached.")
                scalar = scalar.detach().cpu().item()
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                raise TypeError(
                    f"teacher_metrics[{name!r}] must be a detached scalar tensor, int, or float."
                )
            if not math.isfinite(float(scalar)):
                raise ValueError(f"teacher_metrics[{name!r}] must be finite.")
            metrics[name] = scalar
        return metrics

    def _redis_client(self) -> object:
        try:
            import redis
        except ImportError as error:  # pragma: no cover - deployment dependency failure.
            raise RuntimeError("redis-py is required to run the Teacher worker.") from error
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        url = redis_cfg.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("run.redis.url must be a non-empty string.")
        return redis.Redis.from_url(url, decode_responses=False)

    def run_forever(self, stop_event: object) -> None:
        """Consume one bounded upstream turn and publish it downstream."""

        if not callable(getattr(stop_event, "is_set", None)):
            raise TypeError("stop_event must provide is_set().")
        if not self._initialized:
            self.initialize_engine()
        client = self._redis_client()
        keys = self.queue_keys()
        timeout = 1.0
        print(
            f"[Teacher {self.node_id} GPU {self.gpu_id}] Listening on {keys['teacher_train']}; "
            f"max_batches_per_turn={self.max_batches_per_turn}; "
            f"shared_gpu_gate={self.gpu_gate.is_shared}.",
            flush=True,
        )
        while not stop_event.is_set():
            try:
                popped = client.brpop(keys["teacher_train"], timeout=timeout)
            except Exception as error:
                self._publish_event(client, f"Redis receive failed: {error}")
                self._wait_or_stop(stop_event, 1.0)
                continue
            if not popped:
                continue
            _key, payload = popped
            payloads = [payload]
            # Read all immediately available peer batches before taking the
            # physical-GPU permit.  Redis waiting must never block another
            # Worker that already owns the shared card.
            for _ in range(self.max_batches_per_turn - 1):
                try:
                    extra = client.rpop(keys["teacher_train"])
                except Exception:
                    break
                if extra is None:
                    break
                payloads.append(extra)
            wait_started = time.monotonic()
            if not self.gpu_gate.acquire(stop_event):
                # Shutdown won while waiting for a shared GPU.  The payloads
                # stay in this process only, so exit promptly with the Worker.
                break
            gate_wait_seconds = max(0.0, time.monotonic() - wait_started)
            hold_started = time.monotonic()
            try:
                self._activate_model()
                self._process_redis_batches(
                    client,
                    payloads,
                    gate_wait_seconds=gate_wait_seconds,
                )
            finally:
                # CPU offload itself is GPU work and therefore belongs inside
                # the permit.  Its cleanup can still fail (for example after
                # a CUDA error), so releasing the shared semaphore needs its
                # own finally block; otherwise this worker would strand every
                # peer waiting for the same physical GPU.
                try:
                    self._offload_model()
                finally:
                    gate_hold_seconds = max(0.0, time.monotonic() - hold_started)
                    # This both releases the permit and, only for a shared
                    # GPU, yields outside the critical section so another
                    # Worker can take its bounded turn.
                    self.gpu_gate.release_then_yield(stop_event, delay_sec=2.0)
                    print(
                        f"[Teacher {self.node_id} GPU {self.gpu_id}] Completed turn with {len(payloads)} batch(es): "
                        f"gate_wait_seconds={gate_wait_seconds:.3f}, "
                        f"gate_hold_seconds={gate_hold_seconds:.3f}.",
                        flush=True,
                    )

    def _process_redis_batches(
        self,
        client: object,
        payloads: Sequence[object],
        *,
        gate_wait_seconds: float = 0.0,
    ) -> None:
        completed: list[Signals] = []
        for payload in payloads:
            try:
                processing_started = time.monotonic()
                decoded = pickle.loads(payload)
                if not isinstance(decoded, Mapping):
                    raise TypeError("Redis payload did not contain a signal mapping.")
                signals = self.process_training_batch(dict(decoded))
                processing_seconds = max(0.0, time.monotonic() - processing_started)
                completed.append(
                    self._with_processing_timing(
                        signals,
                        processing_seconds,
                        gate_wait_seconds=gate_wait_seconds,
                    )
                )
            except Exception as error:
                self._publish_event(client, f"train batch failed: {error}")
        # GPU kernels from the plugin must complete while the Teacher still
        # owns its physical-GPU permit, and before the resulting signals are
        # made visible to Learner through Redis.
        self._synchronize_device()
        if not completed:
            return
        pipeline = client.pipeline()
        for signals in completed:
            pipeline.lpush(
                self.queue_keys()["learner_train"],
                pickle.dumps(_deep_cpu_detach(signals), protocol=pickle.HIGHEST_PROTOCOL),
            )
        pipeline.execute()

    def _with_processing_timing(
        self,
        signals: Signals,
        processing_seconds: float,
        *,
        gate_wait_seconds: float,
    ) -> Signals:
        """Add Teacher telemetry without replacing upstream Worker timings."""

        timed: Signals = dict(signals)
        raw_timing = timed.get("timing")
        timing: dict[str, object] = dict(raw_timing) if isinstance(raw_timing, Mapping) else {}
        raw_workers = timing.get("workers")
        workers: dict[str, object] = dict(raw_workers) if isinstance(raw_workers, Mapping) else {}
        raw_teacher = workers.get(self.node_id)
        teacher_timing: dict[str, object] = dict(raw_teacher) if isinstance(raw_teacher, Mapping) else {}
        teacher_timing.update(
            {
                "processing_seconds": float(processing_seconds),
                "gate_wait_seconds": max(0.0, float(gate_wait_seconds)),
                "cpu_offload": self.cpu_offload,
                "model_load_seconds": self._model_load_seconds,
                "model_offload_seconds": self._model_offload_seconds,
                "model_load_count": self._model_load_count,
                "model_offload_count": self._model_offload_count,
            }
        )
        workers[self.node_id] = teacher_timing
        timing["workers"] = workers
        timed["timing"] = timing
        metrics_key = f"{self.node_id}_metrics"
        raw_metrics = timed.get(metrics_key)
        metrics: dict[str, object] = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
        metrics.update(
            {
                "gpu_id": self.gpu_id,
                "processing_seconds": float(processing_seconds),
                "gate_wait_seconds": max(0.0, float(gate_wait_seconds)),
                "cpu_offload": self.cpu_offload,
                "model_load_seconds": self._model_load_seconds,
                "model_offload_seconds": self._model_offload_seconds,
                "model_load_count": self._model_load_count,
                "model_offload_count": self._model_offload_count,
            }
        )
        timed[metrics_key] = metrics
        return timed

    def _synchronize_device(self) -> None:
        """Synchronize queued CUDA work before releasing/publishing a turn."""

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _publish_event(self, client: object, message: str) -> None:
        event = {
            "source": self.node_id,
            "gpu_id": self.gpu_id,
            "message": str(message),
            "timestamp": time.time(),
        }
        try:
            client.lpush(
                self.queue_keys()["controller_events"],
                pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL),
            )
        except Exception as error:
            print(f"[Teacher GPU {self.gpu_id}] Could not publish event: {error}", flush=True)

    @staticmethod
    def _wait_or_stop(stop_event: object, seconds: float) -> None:
        wait = getattr(stop_event, "wait", None)
        if callable(wait):
            wait(seconds)
        else:  # pragma: no cover - multiprocessing Event always provides wait.
            time.sleep(seconds)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._model_on_device = False
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


class TeacherManager:
    """Launch one independent no-grad Teacher process per configured GPU."""

    def __init__(
        self,
        config: Mapping[str, object],
        plugin_factory: PluginFactory | None = None,
        *,
        node_id: str = "teacher",
        gpu_gates: GpuGateRegistry | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        self.config = copy.deepcopy(dict(config))
        self.plugin_factory = plugin_factory
        self.node_id = node_id
        if gpu_gates is not None and not isinstance(gpu_gates, GpuGateRegistry):
            raise TypeError("gpu_gates must be a GpuGateRegistry when provided.")
        self.gpu_gates = gpu_gates
        node = resolve_training_pipeline(self.config).node(node_id)
        if node.worker_class != "Teacher":
            raise ValueError(f"Pipeline node {node_id!r} is not a Teacher.")
        teacher_cfg = resolve_worker_config(self.config, node_id)
        run_cfg = _mapping(self.config.get("run"), "run")
        runtime_cfg = _mapping(run_cfg.get("runtime", {}), "run.runtime")
        raw_gpu_ids = teacher_cfg.get("gpu_ids", [])
        if not isinstance(raw_gpu_ids, list):
            raise ValueError(f"{node_id}.gpu_ids must be a list of GPU ids.")
        self.gpu_ids = [self._validated_gpu_id(value) for value in raw_gpu_ids]
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("teacher.gpu_ids may not contain duplicates.")
        if not self.gpu_ids:
            raise ValueError("teacher.gpu_ids must contain at least one GPU id.")
        if float(runtime_cfg.get("minimum_free_gpu_memory_gb", 3)) < 0:
            raise ValueError("run.runtime.minimum_free_gpu_memory_gb must be non-negative.")
        self._context = mp.get_context(str(runtime_cfg.get("process_start_method", "spawn")))
        self.stop_event = self._context.Event()
        self.processes: dict[int, object] = {}
        # Do not replace a Worker that exited while it may have owned a
        # shared-GPU semaphore.  ``multiprocessing.Semaphore`` does not
        # recover a permit from a hard-killed process, so Controller must
        # fail the run rather than start a permanently blocked replacement.
        self.fatal_error: str | None = None
        # See ActorManager.ready_events: Controller waits for this signal
        # before launching the next Worker type on a shared GPU.
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
        print(f"[Teacher Manager {self.node_id}] Started {len(self.processes)} Teacher worker(s).", flush=True)

    def _start_worker(self, gpu_id: int) -> object:
        ready_event = self._context.Event()
        self.ready_events[gpu_id] = ready_event
        gpu_gate = self.gpu_gates.for_gpu(gpu_id) if self.gpu_gates is not None else None
        process = self._context.Process(
            target=_teacher_worker_main,
            args=(gpu_id, self.config, self.plugin_factory, self.stop_event, ready_event, gpu_gate, self.node_id),
            name=f"FrugalOPL-Teacher-{self.node_id}-GPU{gpu_id}",
        )
        process.start()
        return process

    def _uses_shared_gpu_gate(self, gpu_id: int) -> bool:
        """Whether this Teacher's physical GPU has a cross-Worker permit."""

        return self.gpu_gates is not None and self.gpu_gates.for_gpu(gpu_id).is_shared

    def _record_terminal_failure(self, message: str) -> None:
        """Make a Worker exit visible to Controller and stop sibling Teachers."""

        if self.fatal_error is None:
            self.fatal_error = message
            print(f"[Teacher Manager FATAL] {message}", flush=True)
        self.stop_event.set()

    def wait_until_ready(self, *, timeout_sec: float = 300.0) -> None:
        """Block until every initially launched Teacher can score requests."""

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
                        f"Teacher GPU {gpu_id} exited before becoming ready (exitcode={exitcode})."
                    )
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waiting = ", ".join(str(gpu_id) for gpu_id in sorted(pending))
                raise TimeoutError(f"Timed out waiting for Teacher GPU(s) {waiting} to become ready.")
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
                for gpu_id, process in list(self.processes.items()):
                    if process.is_alive():
                        continue
                    exitcode = process.exitcode
                    process.join(timeout=0.1)
                    if self._uses_shared_gpu_gate(gpu_id):
                        self._record_terminal_failure(
                            f"Teacher GPU {gpu_id} exited unexpectedly (exitcode={exitcode}) while using a shared "
                            "GPU gate; automatic restart is unsafe because a hard exit can leave its semaphore "
                            "locked."
                        )
                        return
                    if not restart_enabled:
                        self._record_terminal_failure(
                            f"Teacher GPU {gpu_id} exited unexpectedly (exitcode={exitcode}) and worker restart is "
                            "disabled."
                        )
                        return
                    if self.stop_event.is_set():
                        return
                    print(
                        f"[Teacher Manager] Teacher GPU {gpu_id} exited (exitcode={exitcode}); restarting.",
                        flush=True,
                    )
                    if self.stop_event.wait(delay):
                        break
                    self.processes[gpu_id] = self._start_worker(gpu_id)
                self.stop_event.wait(poll)
        except KeyboardInterrupt:
            self.stop()

    def request_stop(self) -> None:
        """Broadcast cooperative cancellation without joining child processes."""

        self.stop_event.set()

    def _shutdown_grace_sec(self) -> float:
        """Use the longer grace when any Teacher shares its physical GPU."""

        if any(self._uses_shared_gpu_gate(gpu_id) for gpu_id in self.processes):
            return _SHARED_GPU_SHUTDOWN_GRACE_SEC
        return _EXCLUSIVE_GPU_SHUTDOWN_GRACE_SEC

    @staticmethod
    def _join_processes(processes: tuple[object, ...], *, timeout_sec: float) -> None:
        """Join all children against one deadline instead of serial timeouts."""

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
                # Retain a best-effort fallback for process adapters that do
                # not implement ``kill``; never silently drop a child.
                method = getattr(process, "terminate", None)
            if callable(method):
                method()

    def finalize_stop(self) -> None:
        """Let gated CUDA teardown finish before escalating owned children.

        Controller first requests stop for every Worker role.  Thus a Teacher
        blocked behind Actor or Learner's shared-card teardown has already
        been told to exit as soon as the permit becomes available.  Do not
        convert that ordinary wait into a semaphore leak with an early kill.
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
            f"[Teacher Manager] Cooperative shutdown exceeded {grace_sec:.0f}s; "
            f"sending terminate to {len(remaining)} Teacher worker(s).",
            flush=True,
        )
        self._signal_processes(remaining, "terminate")
        self._join_processes(remaining, timeout_sec=_TERMINATE_SHUTDOWN_GRACE_SEC)
        remaining = self._alive_processes(remaining)
        if not remaining:
            return

        print(
            f"[Teacher Manager] Terminate grace exceeded; killing {len(remaining)} Teacher worker(s).",
            flush=True,
        )
        self._signal_processes(remaining, "kill")
        self._join_processes(remaining, timeout_sec=_KILL_SHUTDOWN_JOIN_SEC)
        remaining = self._alive_processes(remaining)
        if remaining:
            raise RuntimeError(
                "Teacher Manager could not reap all owned workers after forced shutdown; "
                f"{len(remaining)} process(es) remain alive."
            )

    def stop(self) -> None:
        """Standalone Manager shutdown, retained for direct Worker CLIs."""

        self.request_stop()
        self.finalize_stop()


def _teacher_worker_main(
    gpu_id: int,
    config: Mapping[str, object],
    plugin_factory: PluginFactory | None,
    stop_event: object,
    ready_event: object,
    gpu_gate: GpuGate | None = None,
    node_id: str = "teacher",
) -> None:
    """Run one independently restartable Teacher process.

    Engine construction is GPU work too: model deserialization may create a
    CUDA context and ``initialize_engine`` eventually migrates the model to
    its device.  It therefore uses the same physical-GPU permit as ordinary
    Teacher turns.  This matters when a restarted Teacher shares a card with
    an already-running Actor or Learner; Controller only orders the initial
    launch and cannot serialize a later Worker retry.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["FRUGALOPL_PHYSICAL_GPU_ID"] = str(gpu_id)

    def request_stop(_signum: int, _frame: object) -> None:
        setter = getattr(stop_event, "set", None)
        if callable(setter):
            setter()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run_cfg = _mapping(config.get("run"), "run")
    restart_cfg = _mapping(_mapping(run_cfg.get("runtime", {}), "run.runtime").get("restart", {}), "run.runtime.restart")
    retry_delay = max(0.0, float(restart_cfg.get("worker_restart_delay_sec", 5.0)))
    # Managers normally inject a registry-owned gate.  Retain a no-op direct
    # construction path for standalone use and CPU tests, while ensuring
    # every object created below sees exactly this same gate.
    initialization_gate = gpu_gate if gpu_gate is not None else GpuGate(physical_gpu_id=gpu_id)

    while not stop_event.is_set():
        teacher: Teacher | None = None
        try:
            print(f"[Teacher {node_id} GPU {gpu_id}] Initializing Transformer model.", flush=True)
            if not initialization_gate.acquire(stop_event):
                # A controlled shutdown won while this Worker waited behind
                # another user of its physical GPU.  Nothing has been
                # allocated in this attempt, so leave promptly.
                return
            try:
                # Construct under the permit as well: device selection can
                # query CUDA, and all later model loading/migration must be
                # mutually exclusive with peers on the physical card.
                teacher_kwargs: dict[str, object] = {"gpu_id": gpu_id, "gpu_gate": initialization_gate}
                if node_id != "teacher":
                    teacher_kwargs["node_id"] = node_id
                teacher = Teacher(config, **teacher_kwargs)  # type: ignore[arg-type]
                teacher.initialize_engine()
                plugin = _build_plugin(plugin_factory, teacher._require_tokenizer())
                if plugin is not None:
                    teacher.set_plugin(plugin)
                offload = getattr(teacher, "_offload_model", None)
                if callable(offload):
                    offload()
            except BaseException as error:
                # If initialization got far enough to allocate CUDA memory,
                # release it before relinquishing the execution permit.  A
                # peer can therefore never issue GPU work concurrently with
                # this failed attempt's cleanup.
                if teacher is not None:
                    teacher.close()
                    teacher = None
                raise
            finally:
                initialization_gate.release()
            print(f"[Teacher {node_id} GPU {gpu_id}] Model ready.", flush=True)
            mark_ready = getattr(ready_event, "set", None)
            if callable(mark_ready):
                mark_ready()
            teacher.run_forever(stop_event)
            return
        except Exception as error:
            print(f"[Teacher GPU {gpu_id} FATAL] {error}\n{traceback.format_exc()}", flush=True)
            Teacher._wait_or_stop(stop_event, retry_delay)
        finally:
            if teacher is not None:
                # ``close`` may empty CUDA's cache, so it is GPU work just
                # like initialization and must not race a shared-card peer.
                initialization_gate.acquire()
                try:
                    teacher.close()
                finally:
                    initialization_gate.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FrugalOPL's generic no-grad Teacher workers.")
    parser.add_argument("--framework-config", required=True, help="Path to a framework profile YAML.")
    parser.add_argument("--algorithm-config", default=None, help="Optional algorithm overlay YAML file.")
    args = parser.parse_args()
    from engine.actor import load_merged_config

    config = load_merged_config(args.framework_config, args.algorithm_config)
    manager = TeacherManager(config, plugin_factory=plugin_factory_from_config(config))
    manager.start()
    manager.join()


if __name__ == "__main__":
    main()


__all__ = ["Teacher", "TeacherManager", "main"]
