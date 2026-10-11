"""A compact RLVR / GRPO-style algorithm plugin.

This module is deliberately an algorithm-layer implementation: it renders
training prompts, turns rollout rewards into group advantages, and defines a
token-level clipped policy objective.  It never sees Redis, a Worker, a CUDA
device, a model object, or gradient-accumulation mechanics.

The plugin is useful as a small, real baseline for checking the full
Controller -> Actor -> Learner training loop.  It is not intended to prescribe
one particular research recipe; more elaborate algorithms can live beside it
without changing the runtime.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

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
from utils.tensor_signals import (
    masked_token_mean,
    require_cpu_vector,
    require_tensor_batch_field,
    require_tensor_signal,
)

from .algorithm_base import AlgorithmPlugin, Metrics, Signals
from .plugin_registry import register_plugin_class

if TYPE_CHECKING:
    from torch import Tensor


_REQUIRED_TRAJECTORY_SIGNALS: dict[str, object] = {
    "responses": True,
    "response_ids": True,
    "prompt_ids": True,
    "prompt_indices": True,
    "response_truncated": True,
    # The sampled-token old-policy logprobs are sufficient for the clipped
    # objective, so no full vocabulary or top-k signal is requested.
    "response_logprobs": True,
}


@register_plugin_class
class RLVRPlugin(AlgorithmPlugin):
    """A minimal executable binary-reward, group-relative RL objective.

    ``collect_learner_signals`` calls the framework answer probe.  Complete
    responses are verified directly; only truncated responses invoke the
    Learner's packed probe forward.  Rewards are then normalized separately
    for every original question.  A group with all correct or all incorrect
    rollouts receives zero advantages, matching the useful GRPO baseline
    behaviour and avoiding an artificial update from no relative signal.
    """

    def __init__(
        self,
        *,
        tokenizer: ChatTemplateTokenizer,
        runtime: AlgorithmRuntimeConfig,
        clip_ratio: float = 0.2,
        answer_probe_threshold: float = 0.0,
    ) -> None:
        if not isinstance(clip_ratio, (int, float)) or not 0.0 < float(clip_ratio) < 1.0:
            raise ValueError("algorithm.clip_ratio must be in (0, 1).")
        if not isinstance(answer_probe_threshold, (int, float)):
            raise TypeError("algorithm.answer_probe_threshold must be numeric.")

        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._clip_ratio = float(clip_ratio)
        self._answer_probe_threshold = float(answer_probe_threshold)
        self._enable_thinking = runtime.enable_thinking

    @classmethod
    def from_runtime_config(
        cls,
        tokenizer: object,
        runtime: AlgorithmRuntimeConfig,
    ) -> "RLVRPlugin":
        """Bind RLVR's private settings to framework-resolved inputs."""

        return cls(
            tokenizer=tokenizer,  # type: ignore[arg-type]
            runtime=runtime,
            clip_ratio=float(runtime.algorithm.get("clip_ratio", 0.2)),
            answer_probe_threshold=float(runtime.algorithm.get("answer_probe_threshold", 0.0)),
        )

    def collect_actor_signals(self, signals: Signals) -> Signals:
        """Render Controller questions and request the compact RLVR rollout."""

        questions, _answers = controller_questions_and_answers(signals)
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
            "return_signals": copy.deepcopy(_REQUIRED_TRAJECTORY_SIGNALS),
        }

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        """RLVR does not require a Teacher Worker."""

        return {}

    def collect_learner_signals(self, signals: Signals) -> Signals:
        """Create complete-batch binary rewards and question-relative advantages."""

        probed = self.answer_probe(
            signals,
            probability_difference_threshold=self._answer_probe_threshold,
        )
        rewards = require_cpu_vector(
            probed.get("rewards"),
            "answer_probe rewards",
            dtype=torch.float32,
        )
        prompt_indices = require_cpu_vector(
            signals.get("prompt_indices"),
            "prompt_indices",
            dtype=torch.long,
        )
        if rewards.numel() != prompt_indices.numel():
            raise RuntimeError("answer_probe rewards and prompt_indices must align by rollout.")

        advantages = torch.zeros_like(rewards)
        for prompt_index in torch.unique(prompt_indices, sorted=True).tolist():
            group = prompt_indices == int(prompt_index)
            group_rewards = rewards[group]
            if group_rewards.numel() < 2:
                # A one-rollout group has no relative training signal.
                continue
            deviation = group_rewards - group_rewards.mean()
            # ``unbiased=False`` is well-defined for small rollout groups.
            scale = group_rewards.std(unbiased=False)
            if float(scale.item()) > 0.0:
                advantages[group] = deviation / scale

        result: Signals = dict(probed)
        result["advantages"] = advantages
        return result

    def compute_objective(self, batch: Signals) -> tuple[Tensor, Metrics]:
        """Return a token-clipped PPO/GRPO objective with maximization semantics."""

        policy = self.compute_logits(
            batch,
            mode="policy_grad",
            return_signals={"response_logprobs": True},
        )
        policy_logprobs = require_tensor_signal(policy, "response_logprobs")
        old_logprobs = require_tensor_batch_field(batch, "response_logprobs", dtype=policy_logprobs.dtype)
        response_mask = require_tensor_batch_field(batch, "response_mask", dtype=torch.bool)
        advantages = require_tensor_batch_field(batch, "advantages", dtype=policy_logprobs.dtype).reshape(-1)

        if policy_logprobs.ndim != 2:
            raise RuntimeError("policy response_logprobs must have shape [batch, response_length].")
        if tuple(old_logprobs.shape) != tuple(policy_logprobs.shape):
            raise RuntimeError("Actor old logprobs must align with policy response logprobs.")
        if tuple(response_mask.shape) != tuple(policy_logprobs.shape):
            raise RuntimeError("response_mask must align with policy response logprobs.")
        if int(advantages.numel()) != int(policy_logprobs.size(0)):
            raise RuntimeError("advantages must contain one value per rollout.")

        valid = response_mask & torch.isfinite(policy_logprobs) & torch.isfinite(old_logprobs)
        safe_delta = torch.where(valid, policy_logprobs - old_logprobs, torch.zeros_like(policy_logprobs))
        # A large numerical discrepancy between vLLM and Transformers should
        # never make the exponential overflow before PPO's normal clipping.
        ratio = safe_delta.clamp(min=-20.0, max=20.0).exp()
        expanded_advantages = advantages.unsqueeze(1)
        unclipped = ratio * expanded_advantages
        clipped = ratio.clamp(1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * expanded_advantages
        surrogate = torch.minimum(unclipped, clipped)
        safe_surrogate = torch.where(valid, surrogate, torch.zeros_like(surrogate))

        token_counts = valid.sum(dim=1).to(dtype=policy_logprobs.dtype).clamp_min(1.0)
        rollout_objectives = safe_surrogate.sum(dim=1) / token_counts
        objective = rollout_objectives.mean()

        clipped_tokens = valid & (ratio.sub(1.0).abs() > self._clip_ratio)
        rewards = require_tensor_batch_field(batch, "rewards", dtype=policy_logprobs.dtype).reshape(-1)
        return objective, {
            "rlvr_reward_mean": rewards.detach().float().mean(),
            "rlvr_advantage_mean": advantages.detach().float().mean(),
            "rlvr_advantage_std": advantages.detach().float().std(unbiased=False),
            "rlvr_loss": -objective.detach(),
            "rlvr_policy_logprob": masked_token_mean(policy_logprobs, valid).detach(),
            "rlvr_ratio_mean": masked_token_mean(ratio, valid).detach(),
            "rlvr_clip_fraction": clipped_tokens.float().sum().detach()
            / valid.float().sum().clamp_min(1.0),
            "rlvr_nonzero_advantage_fraction": (advantages.detach().abs() > 0).float().mean(),
        }

__all__ = ["ChatTemplateTokenizer", "RLVRPlugin"]
