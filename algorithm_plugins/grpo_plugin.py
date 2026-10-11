"""Group-relative clipped policy optimization with answer-probe rewards."""

from __future__ import annotations

import math
import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig, ChatTemplateTokenizer, apply_generation_chat_template,
    controller_questions_and_answers, require_chat_template_tokenizer,
    require_train_template, require_training_sampling,
)
from utils.prompts import render_prompt
from utils.tensor_signals import masked_token_mean, require_cpu_vector

from .algorithm_base import AlgorithmPlugin, Metrics, Signals
from .plugin_registry import register_plugin_class


_TRAJECTORY_SIGNALS = {
    "responses": True, "prompt_ids": True, "response_ids": True,
    "prompt_indices": True, "response_truncated": True, "response_logprobs": True,
}


@register_plugin_class
class GRPOPlugin(AlgorithmPlugin):
    """Maximize the GRPO surrogate minus sampled-token reference KL."""

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        self._clip_ratio = float(runtime.algorithm.get("clip_ratio", 0.2))
        self._kl_coef = float(runtime.algorithm.get("kl_coef", 0.0))
        self._probe_threshold = float(runtime.algorithm.get("answer_probe_threshold", 0.3))
        if not 0.0 < self._clip_ratio < 1.0:
            raise ValueError("algorithm.clip_ratio must be in (0, 1).")
        if not math.isfinite(self._kl_coef) or self._kl_coef < 0:
            raise ValueError("algorithm.kl_coef must be finite and non-negative.")
        if not math.isfinite(self._probe_threshold) or not 0.0 <= self._probe_threshold <= 1.0:
            raise ValueError("algorithm.answer_probe_threshold must be in [0, 1].")

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> "GRPOPlugin":
        return cls(tokenizer=tokenizer, runtime=runtime)

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
        return {"prompts": prompts, "sampling": self._sampling, "return_signals": _TRAJECTORY_SIGNALS}

    def collect_learner_signals(self, signals: Signals) -> Signals:
        probed = self.answer_probe(signals, probability_difference_threshold=self._probe_threshold)
        rewards = require_cpu_vector(probed["rewards"], "rewards", dtype=torch.float32)
        indices = require_cpu_vector(signals["prompt_indices"], "prompt_indices", dtype=torch.long)
        if rewards.numel() != indices.numel():
            raise ValueError("rewards and prompt_indices must align by rollout.")
        advantages = torch.zeros_like(rewards)
        group_means = []
        for question in indices.unique():
            group = indices == question
            values = rewards[group]
            mean = values.mean()
            group_means.append(mean)
            if values.numel() > 1:
                advantages[group] = (values - mean) / (values.std(unbiased=True) + 1e-4)
        means = torch.stack(group_means)
        all_zero = (means == 0).float().mean()
        all_one = (means == 1).float().mean()
        return {**probed, "advantages": advantages, "group_metrics": {
            "grpo_all_zero_group_fraction": all_zero,
            "grpo_all_one_group_fraction": all_one,
            "grpo_zero_advantage_group_fraction": all_zero + all_one,
        }}

    def compute_objective(self, batch: Signals) -> tuple[torch.Tensor, Metrics]:
        reference = self.compute_logits(batch, mode="reference") if self._kl_coef else None
        policy = self.compute_logits(batch, mode="policy_grad", return_signals={
            "response_logprobs": True, "response_entropy": True, "entropy_requires_grad": False,
        })
        logprobs = policy["response_logprobs"]
        old_logprobs = batch["response_logprobs"].detach().to(logprobs)
        response_mask = batch["response_mask"].bool()
        valid = response_mask & torch.isfinite(logprobs) & torch.isfinite(old_logprobs)
        ref_logprobs = reference["response_logprobs"].detach().to(logprobs) if reference is not None else None
        if ref_logprobs is not None:
            valid = valid & torch.isfinite(ref_logprobs)
        delta = torch.where(valid, logprobs - old_logprobs, 0.0)
        ratio = delta.clamp(-20.0, 20.0).exp()
        clip_upper, clip_lower = ratio > 1 + self._clip_ratio, ratio < 1 - self._clip_ratio
        advantages = batch["advantages"].detach().to(logprobs).reshape(-1, 1)
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - self._clip_ratio, 1 + self._clip_ratio) * advantages)
        kl = torch.zeros_like(logprobs)
        if ref_logprobs is not None:
            ref_delta = torch.where(valid, ref_logprobs - logprobs, 0.0).clamp(-20.0, 20.0)
            kl = ref_delta.expm1() - ref_delta  # exp(ref - policy) - (ref - policy) - 1.
        objective = masked_token_mean(surrogate - self._kl_coef * kl, valid)
        return objective, {
            **batch.get("group_metrics", {}),
            "grpo_loss": -objective.detach(),
            "grpo_reward_mean": batch["rewards"].detach().float().mean(),
            "grpo_truncation_rate": batch["response_truncated"].detach().float().mean(),
            "grpo_entropy": masked_token_mean(policy["response_entropy"].detach(), response_mask, finite_only=True),
            "grpo_response_length_mean": response_mask.sum(dim=-1).float().mean(),
            "grpo_kl": masked_token_mean(kl, valid).detach(),
            "grpo_ratio_mean": masked_token_mean(ratio, valid).detach(),
            "grpo_clip_fraction": masked_token_mean((clip_upper | clip_lower).float(), valid).detach(),
            "grpo_clip_upper_fraction": masked_token_mean(clip_upper.float(), valid).detach(),
            "grpo_clip_lower_fraction": masked_token_mean(clip_lower.float(), valid).detach(),
        }


__all__ = ["GRPOPlugin"]
