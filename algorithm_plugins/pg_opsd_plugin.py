"""Policy-gradient self-distillation from a fixed solution-conditioned base model."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig,
    ChatTemplateTokenizer,
    apply_generation_chat_template,
    controller_questions_and_answers,
)
from utils.prompts import OPSD_TEACHER_PROMPT, render_prompt_from_record

from .algorithm_base import Metrics, Signals
from .pg_opd_plugin import PGOPDPlugin, _ACTOR_TRAJECTORY_SIGNALS
from .plugin_registry import register_plugin_class


_LOGPROB_AND_TOPK: dict[str, object] = {
    "response_logprobs": True,
    "topk": {"enabled": True, "top_k": 16},
}


@register_plugin_class
class PGOPSDPlugin(PGOPDPlugin):
    """Use privileged solutions to score the Student's on-policy completions.

    The Teacher is the Learner's frozen base model with LoRA disabled, rather
    than a separately deployed model or the evolving Student policy. Its
    response-aligned scores are collected for the complete logical batch
    before any optimizer update. Only the Student receives a gradient.
    """

    requires_frozen_reference = True

    def __init__(
        self,
        *,
        tokenizer: ChatTemplateTokenizer,
        runtime: AlgorithmRuntimeConfig,
        advantage_clip: float = 10.0,
    ) -> None:
        super().__init__(tokenizer=tokenizer, runtime=runtime, advantage_clip=advantage_clip)
        if not math.isfinite(self._advantage_clip) or self._advantage_clip <= 0:
            raise ValueError("PG-OPSD advantage_clip must be a positive finite number.")

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> "PGOPSDPlugin":
        return cls(
            tokenizer=tokenizer,
            runtime=runtime,
            advantage_clip=runtime.algorithm.get("advantage_clip", 10.0),
        )

    def collect_actor_signals(self, signals: Signals) -> Signals:
        # Fail before generation if a training record lacks privileged text.
        records = self._training_records(signals)
        prompts = [
            apply_generation_chat_template(
                self._tokenizer,
                [{"role": "user", "content": render_prompt_from_record(self._train_template, record)}],
                enable_thinking=self._enable_thinking,
            )
            for record in records
        ]
        return {"prompts": prompts, "sampling": self._sampling, "return_signals": _ACTOR_TRAJECTORY_SIGNALS}

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        del signals
        raise RuntimeError("PG-OPSD collects its fixed Teacher signals in the Learner; omit Teacher nodes.")

    def collect_learner_signals(self, signals: Signals) -> Signals:
        records = self._training_records(signals)
        privileged_prompts = [
            apply_generation_chat_template(
                self._tokenizer,
                [{"role": "user", "content": render_prompt_from_record(OPSD_TEACHER_PROMPT, record)}],
                enable_thinking=self._enable_thinking,
            )
            for record in records
        ]
        traj = signals.get("traj")
        if not isinstance(traj, Mapping):
            raise TypeError("PG-OPSD requires a trajectory mapping at signals['traj'].")
        prompt_indices = traj.get("prompt_indices")
        if (
            not isinstance(prompt_indices, torch.Tensor)
            or prompt_indices.ndim != 1
            or prompt_indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        ):
            raise TypeError("PG-OPSD traj.prompt_indices must be a one-dimensional integer tensor.")
        indices = prompt_indices.tolist()
        if any(index < 0 or index >= len(privileged_prompts) for index in indices):
            raise ValueError("PG-OPSD traj.prompt_indices contains an out-of-range question index.")
        teacher = self.compute_logits(
            signals,
            mode="reference",
            prompts=[privileged_prompts[index] for index in indices],
            return_signals=_LOGPROB_AND_TOPK,
        )
        return {
            "teacher_response_logprobs": teacher["response_logprobs"],
            "teacher_topk_token_ids": teacher["topk_token_ids"],
            "teacher_topk_logprobs": teacher["topk_logprobs"],
        }

    def compute_objective(self, batch: Signals) -> tuple[torch.Tensor, Metrics]:
        objective, metrics = super().compute_objective(batch)
        return objective, {key.replace("pg_opd_", "pg_opsd_", 1): value for key, value in metrics.items()}

    @staticmethod
    def _training_records(signals: Signals) -> list[Mapping[str, object]]:
        questions, _ = controller_questions_and_answers(signals)
        records = signals.get("records")
        if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
            raise ValueError("PG-OPSD requires signals['records'] to contain the original training records.")
        if len(records) != len(questions):
            raise ValueError("PG-OPSD signals['records'] must be aligned with signals['questions'].")
        if not all(isinstance(record.get("solution"), str) and record["solution"].strip() for record in records):
            raise ValueError("PG-OPSD requires a non-empty 'solution' string in every training record.")
        return records


__all__ = ["PGOPSDPlugin"]
