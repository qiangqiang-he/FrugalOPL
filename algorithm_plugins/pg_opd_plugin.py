"""Token-level policy-gradient on-policy distillation with top-16 diagnostics."""

from __future__ import annotations

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig,
    ChatTemplateTokenizer,
    apply_generation_chat_template,
    controller_questions_and_answers,
    require_chat_template_tokenizer,
    require_train_template,
    require_training_sampling,
)
from utils.prompts import render_prompt
from utils.tensor_signals import masked_token_mean

from .algorithm_base import AlgorithmPlugin, Metrics, Signals
from .plugin_registry import register_plugin_class


_ACTOR_TRAJECTORY_SIGNALS: dict[str, object] = {
    "responses": False, "response_ids": True, "prompt_ids": True, "prompt_indices": True,
    "response_truncated": False, "response_logprobs": False,
    "topk": {"enabled": False, "top_k": 0},
}
_TOP_K = 16
_LOGPROB_AND_TOPK: dict[str, object] = {"response_logprobs": True, "topk": {"enabled": True, "top_k": _TOP_K}}


@register_plugin_class
class PGOPDPlugin(AlgorithmPlugin):
    """PG-OPD: maximize detached Teacher-minus-Student token advantages."""

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig, advantage_clip: float = 10.0) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        self._advantage_clip = float(advantage_clip)

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> "PGOPDPlugin":
        return cls(tokenizer=tokenizer, runtime=runtime, advantage_clip=runtime.algorithm.get("advantage_clip", 10.0))

    def collect_actor_signals(self, signals: Signals) -> Signals:
        questions, _ = controller_questions_and_answers(signals)
        prompts = [
            apply_generation_chat_template(
                self._tokenizer,
                [{"role": "user", "content": render_prompt(self._train_template, question=question)}],
                enable_thinking=self._enable_thinking,
            )
            for question in questions
        ]
        return {"prompts": prompts, "sampling": self._sampling, "return_signals": _ACTOR_TRAJECTORY_SIGNALS}

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        questions, _ = controller_questions_and_answers(signals)
        prompts = [
            apply_generation_chat_template(
                self._tokenizer,
                [{"role": "user", "content": render_prompt(self._train_template, question=question)}],
                enable_thinking=self._enable_thinking,
            )
            for question in questions
        ]
        indices = signals["traj"]["prompt_indices"].tolist()
        teacher = self.compute_logits(
            signals,
            prompts=[prompts[index] for index in indices],
            return_signals=_LOGPROB_AND_TOPK,
        )
        return {
            "teacher_response_logprobs": teacher["response_logprobs"],
            "teacher_topk_token_ids": teacher["topk_token_ids"],
            "teacher_topk_logprobs": teacher["topk_logprobs"],
        }

    def collect_learner_signals(self, signals: Signals) -> Signals:
        return {}

    def compute_objective(self, batch: Signals) -> tuple[torch.Tensor, Metrics]:
        policy = self.compute_logits(batch, mode="policy_grad", return_signals=_LOGPROB_AND_TOPK)
        student_logprobs = policy["response_logprobs"]
        teacher_logprobs = batch["teacher_response_logprobs"].to(dtype=student_logprobs.dtype)
        valid = torch.isfinite(student_logprobs) & torch.isfinite(teacher_logprobs)
        raw_advantages = teacher_logprobs - student_logprobs.detach()
        advantages = torch.where(
            valid,
            raw_advantages.clamp(-self._advantage_clip, self._advantage_clip),
            torch.zeros_like(student_logprobs),
        ).detach()
        objective = masked_token_mean(advantages * student_logprobs, valid)

        student_topk_ids, student_topk_logprobs = policy["topk_token_ids"], policy["topk_logprobs"]
        teacher_topk_ids = batch["teacher_topk_token_ids"]
        teacher_topk_logprobs = batch["teacher_topk_logprobs"].to(dtype=student_logprobs.dtype)
        overlap, matched_teacher_logprobs = _match_topk(student_topk_ids, teacher_topk_ids, teacher_topk_logprobs)
        overlap_fraction = overlap.sum(dim=-1).to(dtype=student_logprobs.dtype) / _TOP_K
        student_overlap_mass = torch.where(overlap, student_topk_logprobs.exp(), 0.0).sum(dim=-1)
        symmetric_overlap_mass = torch.where(
            overlap,
            torch.minimum(student_topk_logprobs, matched_teacher_logprobs).exp(),
            0.0,
        ).sum(dim=-1)
        return objective, {
            "pg_opd_loss": -objective.detach(),
            "pg_opd_advantage_mean": masked_token_mean(advantages, valid).detach(),
            "pg_opd_advantage_abs_mean": masked_token_mean(advantages.abs(), valid).detach(),
            "pg_opd_advantage_positive_fraction": masked_token_mean((advantages > 0).float(), valid).detach(),
            "pg_opd_advantage_clipped_fraction": masked_token_mean(
                (raw_advantages.abs() >= self._advantage_clip).float(), valid
            ).detach(),
            "pg_opd_top_16_overlap": masked_token_mean(overlap_fraction, valid).detach(),
            "pg_opd_top_16_student_overlap_mass": masked_token_mean(student_overlap_mass, valid).detach(),
            "pg_opd_top_16_symmetric_overlap_mass": masked_token_mean(symmetric_overlap_mass, valid).detach(),
            "pg_opd_teacher_prob": masked_token_mean(teacher_logprobs.exp(), valid).detach(),
            "pg_opd_student_prob": masked_token_mean(student_logprobs.exp(), valid).detach(),
        }


def _match_topk(student_ids: torch.Tensor, teacher_ids: torch.Tensor, teacher_logprobs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Teacher scores without a ``[1, L, 16, 16]`` float temporary."""

    overlap = torch.zeros_like(student_ids, dtype=torch.bool)
    matched_teacher_logprobs = torch.full_like(teacher_logprobs, -torch.inf)
    for index in range(_TOP_K):
        teacher_id = teacher_ids[..., index].unsqueeze(-1)
        matches = student_ids.eq(teacher_id)
        overlap |= matches
        matched_teacher_logprobs = torch.where(matches, teacher_logprobs[..., index].unsqueeze(-1), matched_teacher_logprobs)
    return overlap, matched_teacher_logprobs


__all__ = ["ChatTemplateTokenizer", "PGOPDPlugin"]
