"""Policy-gradient self-distillation from a fixed solution-conditioned base model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig,
    ChatTemplateTokenizer,
    apply_generation_chat_template,
    controller_questions_and_answers,
)
from utils.prompts import OPSD_TEACHER_PROMPT, render_prompt_from_record

from .algorithm_base import ObjectiveResult, Signals
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
        self._max_prompt_length = runtime.max_prompt_length

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
            self._render_student_prompt(record)
            for record in records
        ]
        return {"prompts": prompts, "sampling": self._sampling, "return_signals": _ACTOR_TRAJECTORY_SIGNALS}

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        del signals
        raise RuntimeError("PG-OPSD collects its fixed Teacher signals in the Learner; omit Teacher nodes.")

    def collect_learner_signals(self, signals: Signals) -> Signals:
        records = self._training_records(signals)
        privileged_prompts = [
            self._render_privileged_prompt(record)
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

    def _render_student_prompt(self, record: Mapping[str, object]) -> str:
        """Render a student prompt and reject an overlong question explicitly.

        PG-OPSD may safely shorten only the privileged reference solution.
        Cutting a student question would silently change the task, so surface
        that configuration/data mismatch before Actor generation instead.
        """

        prompt = apply_generation_chat_template(
            self._tokenizer,
            [{"role": "user", "content": render_prompt_from_record(self._train_template, record)}],
            enable_thinking=self._enable_thinking,
        )
        self._assert_prompt_within_limit(prompt, kind="student")
        return prompt

    def _render_privileged_prompt(self, record: Mapping[str, object]) -> str:
        """Render the Teacher prompt, shortening only ``solution`` if needed.

        The final rendered chat string is token-counted on every candidate.
        This avoids truncating the generation suffix or depending on BPE
        boundaries at the template/solution join.
        """

        prompt = self._render_teacher_record(record)
        limit = self._max_prompt_length
        if limit is None or self._prompt_token_count(prompt) <= limit:
            return prompt

        solution = record.get("solution")
        if not isinstance(solution, str) or not solution:
            # _training_records already validates this; retain a local guard
            # so this helper cannot accidentally render an invalid record.
            raise ValueError("PG-OPSD requires a non-empty 'solution' string in every training record.")

        immutable_record = dict(record)
        immutable_record["solution"] = ""
        immutable_prompt = self._render_teacher_record(immutable_record)
        immutable_tokens = self._prompt_token_count(immutable_prompt)
        if immutable_tokens > limit:
            raise ValueError(
                "PG-OPSD's question/template exceeds controller.data.max_prompt_length "
                f"({immutable_tokens} > {limit}); refusing to truncate the question."
            )

        # Keep a prefix of the solution.  A final token-count check is made
        # for each candidate because tokenization at the solution boundary is
        # model-specific.  The rendered prompt retains both the fixed prompt
        # suffix and the assistant generation prefix.
        lower, upper = 0, len(solution)
        best_prompt = immutable_prompt
        while lower <= upper:
            midpoint = (lower + upper) // 2
            candidate_record = dict(record)
            candidate_record["solution"] = solution[:midpoint]
            candidate_prompt = self._render_teacher_record(candidate_record)
            if self._prompt_token_count(candidate_prompt) <= limit:
                best_prompt = candidate_prompt
                lower = midpoint + 1
            else:
                upper = midpoint - 1

        # The binary search above always starts from an already-verified
        # empty-solution prompt.  Keep this assertion as a defensive guard
        # against a tokenizer with unusual non-monotonic segmentation.
        if self._prompt_token_count(best_prompt) > limit:  # pragma: no cover - defensive invariant.
            raise RuntimeError("PG-OPSD failed to construct a prompt within its configured token limit.")
        return best_prompt

    def _render_teacher_record(self, record: Mapping[str, object]) -> str:
        return apply_generation_chat_template(
            self._tokenizer,
            [{"role": "user", "content": render_prompt_from_record(OPSD_TEACHER_PROMPT, record)}],
            enable_thinking=self._enable_thinking,
        )

    def _assert_prompt_within_limit(self, prompt: str, *, kind: str) -> None:
        if self._max_prompt_length is None:
            return
        token_count = self._prompt_token_count(prompt)
        if token_count > self._max_prompt_length:
            raise ValueError(
                f"PG-OPSD {kind} prompt exceeds controller.data.max_prompt_length "
                f"({token_count} > {self._max_prompt_length})."
            )

    def _prompt_token_count(self, prompt: str) -> int:
        tokenizer_call = getattr(self._tokenizer, "__call__", None)
        if not callable(tokenizer_call):
            raise TypeError("PG-OPSD prompt limiting requires a callable tokenizer.")
        encoded = tokenizer_call(prompt, add_special_tokens=False)
        if not isinstance(encoded, Mapping):
            raise TypeError("Tokenizer prompt output must be a mapping.")
        raw_ids = encoded.get("input_ids")
        to_list = getattr(raw_ids, "tolist", None)
        if callable(to_list):
            raw_ids = to_list()
        if isinstance(raw_ids, Sequence) and raw_ids and isinstance(raw_ids[0], Sequence) and not isinstance(
            raw_ids[0], (str, bytes)
        ):
            if len(raw_ids) != 1:
                raise ValueError("Tokenizer returned multiple prompt token rows.")
            raw_ids = raw_ids[0]
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise TypeError("Tokenizer prompt output input_ids must be a token sequence.")
        return len(raw_ids)

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        result = super().compute_objective(batch)
        return ObjectiveResult(
            objective=result.objective,
            metrics={key.replace("pg_opd_", "pg_opsd_", 1): value for key, value in result.metrics.items()},
            valid_token_count=result.valid_token_count,
        )

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
