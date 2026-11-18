"""Group-relative clipped policy optimization with answer-probe rewards.

This is also the minimal binary-reward baseline for checking the complete
Controller -> Actor -> Learner loop. The plugin owns prompts, group-relative
advantages, and the objective; the framework owns models and execution.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig, ChatTemplateTokenizer, apply_generation_chat_template,
    controller_questions_and_answers, require_chat_template_tokenizer,
    require_train_template, require_training_sampling,
)
from utils.prompts import render_prompt
from utils.tensor_signals import (
    masked_token_mean, require_cpu_vector, require_tensor_batch_field,
    require_tensor_signal,
)

from .algorithm_base import AlgorithmPlugin, ObjectiveResult, Signals
from .dynamic_advantage import DynamicAdvantage, DynamicAdvantageConfig
from .plugin_registry import register_plugin_class
from .qae import empirical_quantile_baseline, validate_qae_quantile


_TRAJECTORY_SIGNALS = {
    "responses": True, "prompt_ids": True, "response_ids": True,
    "prompt_indices": True, "response_truncated": True, "response_logprobs": True,
}


def _epsilon(value: object, name: str, *, below_one: bool = False) -> float:
    constraint = "finite and in (0, 1)" if below_one else "finite and positive"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"algorithm.{name} must be a number that is {constraint}.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"algorithm.{name} must be a number that is {constraint}.") from error
    if not math.isfinite(result) or result <= 0 or (below_one and result >= 1):
        raise ValueError(f"algorithm.{name} must be a number that is {constraint}.")
    return result


def _advantage_scale(value: object, name: str) -> float:
    message = f"algorithm.{name} must be a finite non-negative number."
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(message)
    return result


@register_plugin_class
class GRPOPlugin(AlgorithmPlugin):
    """Maximize the GRPO surrogate minus optional sampled-token reference KL.

    Complete responses receive verified binary rewards; truncated responses
    use the Learner's answer probe. Single-rollout and constant-reward groups
    have zero advantages. No Teacher Worker is needed, including for KL,
    whose frozen reference is scored by the Learner.
    """

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        self._advantage_estimator = runtime.algorithm.get("advantage_estimator", "grpo")
        if not isinstance(self._advantage_estimator, str) or self._advantage_estimator not in {"grpo", "qae"}:
            raise ValueError("algorithm.advantage_estimator must be 'grpo' or 'qae'.")
        self._qae_quantile = (
            validate_qae_quantile(runtime.algorithm.get("qae_quantile", 0.4))
            if self._advantage_estimator == "qae" else None
        )
        if self._advantage_estimator != "qae" and "qae_quantile" in runtime.algorithm:
            raise ValueError("algorithm.qae_quantile requires algorithm.advantage_estimator='qae'.")
        if "clip_ratio" in runtime.algorithm:
            raise ValueError("algorithm.clip_ratio is unsupported; use algorithm.epsilon_low and algorithm.epsilon_high.")
        self._epsilon_low = _epsilon(runtime.algorithm.get("epsilon_low", 0.2), "epsilon_low", below_one=True)
        self._epsilon_high = _epsilon(runtime.algorithm.get("epsilon_high", 0.2), "epsilon_high")
        self._positive_advantage_scale = _advantage_scale(
            runtime.algorithm.get("positive_advantage_scale", 1.0), "positive_advantage_scale",
        )
        self._negative_advantage_scale = _advantage_scale(
            runtime.algorithm.get("negative_advantage_scale", 1.0), "negative_advantage_scale",
        )
        dynamic_config = DynamicAdvantageConfig.from_mapping(runtime.algorithm.get("dynamic_advantage", {}))
        if self._advantage_estimator == "qae" and (
            dynamic_config.enabled
            or self._positive_advantage_scale != 1.0
            or self._negative_advantage_scale != 1.0
        ):
            raise ValueError(
                "QAE requires dynamic_advantage.enabled=false and "
                "positive_advantage_scale=negative_advantage_scale=1.0."
            )
        if dynamic_config.enabled and (
            self._positive_advantage_scale != 1.0 or self._negative_advantage_scale != 1.0
        ):
            raise ValueError("Dynamic advantage requires positive_advantage_scale=negative_advantage_scale=1.0.")
        self._dynamic_advantage = DynamicAdvantage(dynamic_config)
        self._kl_coef = float(runtime.algorithm.get("kl_coef", 0.0))
        self._probe_threshold = float(runtime.algorithm.get("answer_probe_threshold", 0.3))
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
        return {
            "prompts": prompts,
            "sampling": copy.deepcopy(self._sampling),
            "return_signals": copy.deepcopy(_TRAJECTORY_SIGNALS),
        }

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        """GRPO obtains rewards and optional reference scores in the Learner."""

        return {}

    def on_training_step_completed(self, *, step: int, metrics: Mapping[str, object]) -> dict[str, object]:
        if not self._dynamic_advantage.config.enabled:
            return {}
        if "grpo_entropy" not in metrics:
            raise ValueError("Dynamic advantage requires a globally aggregated grpo_entropy metric.")
        return self._dynamic_advantage.observe(step=step, entropy=metrics["grpo_entropy"])

    def training_state_dict(self) -> dict[str, object]:
        if self._advantage_estimator == "qae":
            # QAE has no running statistics, but the estimator and quantile
            # must not change silently when an optimizer checkpoint resumes.
            return {"version": 1, "advantage_estimator": "qae", "qae_quantile": self._qae_quantile}
        return self._dynamic_advantage.state_dict()

    def load_training_state_dict(self, state: Mapping[str, object], *, step: int) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Algorithm checkpoint step must be a non-negative integer.")
        if not isinstance(state, Mapping):
            raise ValueError("Algorithm checkpoint state must be a mapping.")
        if self._advantage_estimator == "qae":
            if not state and step == 0:
                return
            if state.get("advantage_estimator") != "qae":
                raise ValueError("QAE cannot resume a checkpoint from a different advantage estimator.")
            if set(state) != {"version", "advantage_estimator", "qae_quantile"} or (
                type(state["version"]) is not int or state["version"] != 1
            ):
                raise ValueError("QAE checkpoint has missing or unsupported state fields/version.")
            quantile = validate_qae_quantile(state["qae_quantile"])
            if quantile != self._qae_quantile:
                raise ValueError("QAE checkpoint quantile does not match algorithm.qae_quantile.")
            return
        if "advantage_estimator" in state:
            raise ValueError("GRPO cannot resume a checkpoint from a different advantage estimator.")
        # Preserve the original flat dynamic-advantage checkpoint format and
        # legacy empty state for standard GRPO.
        self._dynamic_advantage.load_state_dict(state, step=step)

    def collect_learner_signals(self, signals: Signals) -> Signals:
        probed = self.answer_probe(signals, probability_difference_threshold=self._probe_threshold)
        rewards = require_cpu_vector(probed["rewards"], "rewards", dtype=torch.float32)
        indices = require_cpu_vector(signals["prompt_indices"], "prompt_indices", dtype=torch.long)
        if rewards.numel() != indices.numel():
            raise ValueError("rewards and prompt_indices must align by rollout.")
        if rewards.numel() == 0:
            raise ValueError("GRPO requires at least one rollout to compute advantages.")
        advantages = torch.zeros_like(rewards)
        group_means = []
        group_baselines = []
        for question in indices.unique():
            group = indices == question
            values = rewards[group]
            mean = values.mean()
            group_means.append(mean)
            baseline = (
                empirical_quantile_baseline(values, self._qae_quantile)
                if self._advantage_estimator == "qae" else mean
            )
            group_baselines.append(baseline)
            if values.numel() > 1:
                # QAE changes only the baseline. Keep GRPO's original reward
                # standard deviation and full group/rollout denominators;
                # neither re-center nor normalize by nonzero advantages.
                advantages[group] = (values - baseline) / (values.std(unbiased=True) + 1e-4)
        negative_scale = (
            self._dynamic_advantage.alpha_neg
            if self._dynamic_advantage.config.enabled else self._negative_advantage_scale
        )
        if self._positive_advantage_scale != 1.0 or negative_scale != 1.0:
            # Scale only after every group's standard GRPO advantages exist.
            # Keep their original signs and zeros; do not normalize again.
            positive = advantages > 0
            negative = advantages < 0
            advantages[positive] *= self._positive_advantage_scale
            advantages[negative] *= negative_scale
        means = torch.stack(group_means)
        baselines = torch.stack(group_baselines)
        all_zero = (means == 0).float().mean()
        all_one = (means == 1).float().mean()
        return {**probed, "advantages": advantages, "group_metrics": {
            "grpo_all_zero_group_fraction": all_zero,
            "grpo_all_one_group_fraction": all_one,
            "grpo_zero_advantage_group_fraction": all_zero + all_one,
            # The objective sees one rollout, so distribution statistics must
            # be computed here while the complete logical batch is available.
            "grpo_advantage_mean": advantages.mean(),
            "grpo_advantage_std": advantages.std(unbiased=False),
            "grpo_advantage_baseline_mean": baselines.mean(),
            "grpo_advantage_estimator_qae": float(self._advantage_estimator == "qae"),
            "grpo_positive_advantage_fraction": (advantages > 0).float().mean(),
            "grpo_negative_advantage_fraction": (advantages < 0).float().mean(),
            "grpo_nonzero_advantage_fraction": (advantages.abs() > 0).float().mean(),
            **({
                "grpo_qae_quantile": self._qae_quantile,
                "grpo_qae_baseline_mean": baselines.mean(),
            } if self._advantage_estimator == "qae" else {}),
        }}

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        reference = self.compute_logits(batch, mode="reference") if self._kl_coef else None
        policy = self.compute_logits(batch, mode="policy_grad", return_signals={
            "response_logprobs": True, "response_entropy": True, "entropy_requires_grad": False,
        })
        logprobs = require_tensor_signal(policy, "response_logprobs")
        entropy = require_tensor_signal(policy, "response_entropy")
        old_logprobs = require_tensor_batch_field(batch, "response_logprobs").detach().to(logprobs)
        response_mask = require_tensor_batch_field(batch, "response_mask", dtype=torch.bool)
        advantages = require_tensor_batch_field(batch, "advantages").detach().to(logprobs).reshape(-1, 1)
        rewards = require_tensor_batch_field(batch, "rewards").detach().float().reshape(-1)
        truncated = require_tensor_batch_field(batch, "response_truncated").detach().float().reshape(-1)
        if logprobs.ndim != 2:
            raise RuntimeError("policy response_logprobs must have shape [batch, response_length].")
        for name, value in (
            ("Actor old logprobs", old_logprobs), ("response_mask", response_mask),
            ("policy response_entropy", entropy),
        ):
            if tuple(value.shape) != tuple(logprobs.shape):
                raise RuntimeError(f"{name} must align with policy response logprobs.")
        for name, value in (("advantages", advantages), ("rewards", rewards), ("response_truncated", truncated)):
            if value.numel() != logprobs.size(0):
                raise RuntimeError(f"{name} must contain one value per rollout.")
        valid = response_mask & torch.isfinite(logprobs) & torch.isfinite(old_logprobs)
        ref_logprobs = (
            require_tensor_signal(reference, "response_logprobs").detach().to(logprobs)
            if reference is not None else None
        )
        if ref_logprobs is not None:
            if tuple(ref_logprobs.shape) != tuple(logprobs.shape):
                raise RuntimeError("reference response_logprobs must align with policy response logprobs.")
            valid = valid & torch.isfinite(ref_logprobs)
        delta = torch.where(valid, logprobs - old_logprobs, 0.0)
        ratio = delta.clamp(-20.0, 20.0).exp()
        lower_bound, upper_bound = 1 - self._epsilon_low, 1 + self._epsilon_high
        clip_upper, clip_lower = ratio > upper_bound, ratio < lower_bound
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(lower_bound, upper_bound) * advantages)
        kl = torch.zeros_like(logprobs)
        if ref_logprobs is not None:
            ref_delta = torch.where(valid, ref_logprobs - logprobs, 0.0).clamp(-20.0, 20.0)
            kl = ref_delta.expm1() - ref_delta  # exp(ref - policy) - (ref - policy) - 1.
        objective = masked_token_mean(surrogate - self._kl_coef * kl, valid)
        return ObjectiveResult(objective=objective, metrics={
            **batch.get("group_metrics", {}),
            "grpo_loss": -objective.detach(),
            "grpo_reward_mean": rewards.mean(),
            "grpo_policy_logprob": masked_token_mean(logprobs, valid).detach(),
            "grpo_truncation_rate": truncated.mean(),
            "grpo_entropy": masked_token_mean(entropy.detach(), response_mask, finite_only=True),
            "grpo_response_length_mean": response_mask.sum(dim=-1).float().mean(),
            "grpo_kl": masked_token_mean(kl, valid).detach(),
            "grpo_ratio_mean": masked_token_mean(ratio, valid).detach(),
            "grpo_clip_fraction": masked_token_mean((clip_upper | clip_lower).float(), valid).detach(),
            "grpo_clip_upper_fraction": masked_token_mean(clip_upper.float(), valid).detach(),
            "grpo_clip_lower_fraction": masked_token_mean(clip_lower.float(), valid).detach(),
        }, valid_token_count=valid.sum().detach())


__all__ = ["GRPOPlugin"]
