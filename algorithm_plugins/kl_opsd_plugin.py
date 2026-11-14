"""On-policy self-distillation using the official full-vocabulary Forward KL."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from utils.algorithm_runtime import AlgorithmRuntimeConfig, ChatTemplateTokenizer, apply_generation_chat_template
from utils.prompts import KL_OPSD_TEACHER_PROMPT, render_prompt_from_record
from utils.tensor_signals import masked_token_mean

from .algorithm_base import ObjectiveResult, Signals
from .pg_opsd_plugin import PGOPSDPlugin
from .plugin_registry import register_plugin_class


@register_plugin_class
class KLOPSDPlugin(PGOPSDPlugin):
    """Match a frozen, solution-conditioned Teacher on Student trajectories.

    Only prompt construction/validation is shared with PG-OPSD. There are no
    advantages, rewards, sampled-token KL estimates, or top-k approximations
    in this objective. Reference logits are computed locally during each
    Learner objective, with LoRA disabled and without Teacher gradients.
    """

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig) -> None:
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        if runtime.algorithm.get("loss_aggregation") != "token":
            raise ValueError("KL-OPSD requires algorithm.loss_aggregation='token' to match upstream batchmean.")
        self._kl_temperature = self._positive_number(
            runtime.algorithm.get("kl_temperature", self._sampling.get("temperature", 1.0)), "kl_temperature",
        )
        clip = runtime.algorithm.get("kl_token_clip", 1.0e-7)
        self._kl_token_clip = None if clip is None else self._positive_number(clip, "kl_token_clip")
        chunk_size = runtime.algorithm.get("kl_chunk_size", 128)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("KL-OPSD kl_chunk_size must be a positive integer.")
        self._kl_chunk_size = chunk_size

    @staticmethod
    def _positive_number(value: object, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"KL-OPSD {name} must be a positive finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"KL-OPSD {name} must be a positive finite number.") from error
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"KL-OPSD {name} must be a positive finite number.")
        return number

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> "KLOPSDPlugin":
        return cls(tokenizer=tokenizer, runtime=runtime)

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        del signals
        raise RuntimeError("KL-OPSD uses the Learner's frozen reference; omit separate Teacher nodes.")

    def collect_learner_signals(self, signals: Signals) -> Signals:
        records = self._training_records(signals)
        prompts = [self._render_privileged_prompt(record) for record in records]
        traj = signals.get("traj")
        if not isinstance(traj, Mapping):
            raise TypeError("KL-OPSD requires a trajectory mapping at signals['traj'].")
        indices = traj.get("prompt_indices")
        if (
            not isinstance(indices, torch.Tensor) or indices.ndim != 1
            or indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        ):
            raise TypeError("KL-OPSD traj.prompt_indices must be a one-dimensional integer tensor.")
        indices = indices.tolist()
        if any(index < 0 or index >= len(prompts) for index in indices):
            raise ValueError("KL-OPSD traj.prompt_indices contains an out-of-range question index.")
        # Keep only prompt strings in the logical batch. Full-vocabulary
        # tensors must never enter CPU signal caches or distributed gathers.
        return {"teacher_prompts": [prompts[index] for index in indices]}

    def _render_teacher_record(self, record: Mapping[str, object]) -> str:
        return apply_generation_chat_template(
            self._tokenizer,
            [{"role": "user", "content": render_prompt_from_record(KL_OPSD_TEACHER_PROMPT, record)}],
            enable_thinking=self._enable_thinking,
        )

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        values = self.compute_forward_kl(
            batch, teacher_prompts=batch.get("teacher_prompts"),
            temperature=self._kl_temperature, token_clip=self._kl_token_clip, chunk_size=self._kl_chunk_size,
        )
        mask = batch.get("response_mask")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("KL-OPSD requires a tensor response_mask.")
        mask = mask.bool()
        loss = masked_token_mean(values["response_forward_kl"], mask)
        return ObjectiveResult(
            # FrugalOPL maximizes objectives; upstream minimizes the KL loss.
            objective=-loss,
            metrics={
                "kl_opsd_loss": loss.detach(),
                "kl_opsd_unclipped_kl": masked_token_mean(values["response_forward_kl_unclipped"], mask).detach(),
                "kl_opsd_clip_fraction": masked_token_mean(values["response_forward_kl_clipped_fraction"], mask).detach(),
            },
            valid_token_count=mask.sum().detach(),
        )

    def training_state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "direction": "forward",
            "temperature": self._kl_temperature,
            "token_clip": self._kl_token_clip,
            "loss_aggregation": "token",
        }

    def load_training_state_dict(self, state: Mapping[str, object], *, step: int) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("KL-OPSD saved algorithm state must be a mapping.")
        if not state and step == 0:
            return
        expected = self.training_state_dict()
        if set(state) != set(expected) or any(
            type(state[key]) is not type(value) or state[key] != value for key, value in expected.items()
        ):
            raise ValueError("KL-OPSD checkpoint objective settings do not match this run.")


__all__ = ["KLOPSDPlugin"]
