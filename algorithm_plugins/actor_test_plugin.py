"""A small executable Actor-to-Learner smoke algorithm.

``ActorTestPlugin`` is intentionally not a research algorithm.  It provides
one real, minimal plugin that exercises the public framework boundary end to
end: Actor renders no-thinking prompts and produces trajectories; Learner
uses the answer probe, obtains all three policy/reference signal modes, and
returns a differentiable objective.  This makes it suitable for validating a
new model/runtime without putting framework details in algorithm code.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
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
    require_tensor_batch_field,
    require_tensor_signal,
)

from .algorithm_base import AlgorithmPlugin, Metrics, Signals
from .plugin_registry import register_plugin_class

if TYPE_CHECKING:
    from torch import Tensor


_SMOKE_TRAJECTORY_SIGNALS: dict[str, object] = {
    "responses": True,
    "response_ids": True,
    "prompt_ids": True,
    "prompt_indices": True,
    "response_truncated": True,
    "response_logprobs": True,
}

_SMOKE_METRIC_MASK_SHAPE_ERROR = "Token signal and response mask must have the same shape."


@register_plugin_class
class ActorTestPlugin(AlgorithmPlugin):
    """Exercise the generic Actor and Learner contracts with real tensors.

    Args:
        tokenizer: The Actor tokenizer.  It is injected by the Worker so the
            plugin always uses the same chat template as vLLM.
        runtime: Framework-resolved prompt, sampling, and thinking inputs.
            The plugin only sees the algorithm-scoped runtime snapshot, never
            the merged training configuration.
        return_signals: Optional rollout signal requirements.  The fields
            required by the framework answer probe are always requested.
        answer_probe_threshold: A truncated rollout is rewarded when its end
            answer-probe probability exceeds its start probability by more
            than this value.  ``-1`` is convenient for an API smoke test.
    """

    def __init__(
        self,
        *,
        tokenizer: ChatTemplateTokenizer,
        runtime: AlgorithmRuntimeConfig,
        return_signals: Mapping[str, object] | None = None,
        answer_probe_threshold: float = -1.0,
    ) -> None:
        if return_signals is not None and not isinstance(return_signals, Mapping):
            raise TypeError("return_signals must be a mapping when provided.")
        if not isinstance(answer_probe_threshold, (int, float)):
            raise TypeError("answer_probe_threshold must be numeric.")

        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._return_signals: Signals = copy.deepcopy(_SMOKE_TRAJECTORY_SIGNALS)
        if return_signals is not None:
            self._return_signals.update(copy.deepcopy(dict(return_signals)))
        missing = [
            key for key, expected in _SMOKE_TRAJECTORY_SIGNALS.items() if self._return_signals.get(key) is not expected
        ]
        if missing:
            raise ValueError(
                "ActorTestPlugin needs the complete trajectory for Learner answer-probe testing; "
                f"these return_signals must be true: {missing}."
        )
        self._answer_probe_threshold = float(answer_probe_threshold)
        self._enable_thinking = runtime.enable_thinking
        self._alternate_prompt = self._render_alternate_prompt()

    @classmethod
    def from_runtime_config(
        cls,
        tokenizer: object,
        runtime: AlgorithmRuntimeConfig,
    ) -> "ActorTestPlugin":
        """Bind smoke-plugin settings to framework-resolved inputs."""

        return_signals = runtime.algorithm.get("return_signals")
        return cls(
            tokenizer=tokenizer,  # type: ignore[arg-type]
            runtime=runtime,
            return_signals=return_signals,
            answer_probe_threshold=float(runtime.algorithm.get("answer_probe_threshold", -1.0)),
        )

    def collect_actor_signals(self, signals: Signals) -> Signals:
        """Render ``questions`` as user messages and request rollout signals.

        ``answers`` is kept as opaque Controller data because it travels with
        the batch for downstream reward logic, but this prompt-only plugin
        does not inspect or modify it.  The returned mapping contains only fields newly
        produced by Actor-side algorithm logic; the engine keeps the original
        ``questions`` and ``answers`` when it merges the result.
        """

        questions, _answers = controller_questions_and_answers(signals)
        prompts: list[str] = []
        for question in questions:
            user_text = render_prompt(self._train_template, question=question)
            prompts.append(
                apply_generation_chat_template(
                    self._tokenizer,
                    [{"role": "user", "content": user_text}],
                    enable_thinking=self._enable_thinking,
                )
            )

        return {
            "prompts": prompts,
            "sampling": copy.deepcopy(self._sampling),
            "return_signals": copy.deepcopy(self._return_signals),
        }

    def collect_teacher_signals(self, signals: Signals) -> Signals:
        """The smoke plugin intentionally does not request a Teacher."""

        return {}

    def collect_learner_signals(self, signals: Signals) -> Signals:
        """Compute framework answer-probe rewards for the full rollout group."""

        return self.answer_probe(
            signals,
            probability_difference_threshold=self._answer_probe_threshold,
        )

    def compute_objective(self, batch: Signals) -> tuple[Tensor, Metrics]:
        """Exercise all Learner model-signal modes with a real objective.

        The objective is deliberately simple and not presented as a research
        loss: reward ``1`` maximizes the sampled-response likelihood, while
        reward ``0`` minimizes it.  Its only purpose is to verify that a
        plugin can combine detached rollout signals, no-grad reference
        signals, and a gradient-bearing policy signal without touching model,
        device, gradient accumulation, or distributed state.
        """

        reference = self.compute_logits(
            batch,
            mode="reference",
            return_signals={"response_logprobs": True},
        )
        policy_no_grad = self.compute_logits(
            batch,
            mode="policy_no_grad",
            return_signals={
                "response_logprobs": True,
                "response_entropy": True,
                "topk": {"enabled": True, "top_k": 4},
            },
        )
        # These two calls explicitly exercise the OPSD-style prompt override
        # paths.  A Learner objective always receives one rollout, so the
        # string override has one item and the token-ID override is already
        # the exact unpadded ``[1, L]`` prompt tensor.
        alternate_prompt = self.compute_logits(
            batch,
            mode="policy_no_grad",
            prompts=[self._alternate_prompt],
            return_signals={"response_logprobs": True},
        )
        prompt_id_override = self.compute_logits(
            batch,
            mode="policy_no_grad",
            prompt_ids=require_tensor_batch_field(batch, "prompt_ids"),
            return_signals={"response_logprobs": True},
        )
        policy = self.compute_logits(
            batch,
            mode="policy_grad",
            return_signals={"response_logprobs": True},
        )

        reference_logprobs = require_tensor_signal(reference, "response_logprobs")
        no_grad_logprobs = require_tensor_signal(policy_no_grad, "response_logprobs")
        alternate_logprobs = require_tensor_signal(alternate_prompt, "response_logprobs")
        prompt_id_logprobs = require_tensor_signal(prompt_id_override, "response_logprobs")
        policy_logprobs = require_tensor_signal(policy, "response_logprobs")
        entropy = require_tensor_signal(policy_no_grad, "response_entropy")
        topk_ids = require_tensor_signal(policy_no_grad, "topk_token_ids")
        topk_logprobs = require_tensor_signal(policy_no_grad, "topk_logprobs")
        response_mask = require_tensor_batch_field(batch, "response_mask", dtype=policy_logprobs.dtype)
        rewards = require_tensor_batch_field(batch, "rewards", dtype=policy_logprobs.dtype).reshape(-1)

        if tuple(reference_logprobs.shape) != tuple(policy_logprobs.shape):
            raise RuntimeError("Reference and policy response-logprob shapes differ.")
        if tuple(entropy.shape) != tuple(policy_logprobs.shape):
            raise RuntimeError("Response entropy must align with policy response logprobs.")
        if tuple(topk_ids.shape[:2]) != tuple(policy_logprobs.shape) or tuple(topk_logprobs.shape[:2]) != tuple(
            policy_logprobs.shape
        ):
            raise RuntimeError("Top-k response signals must align with policy response logprobs.")
        if reference_logprobs.requires_grad or no_grad_logprobs.requires_grad:
            raise RuntimeError("Reference and policy_no_grad signals must be detached.")
        if not policy_logprobs.requires_grad:
            raise RuntimeError("policy_grad response logprobs must retain a gradient graph.")
        if tuple(response_mask.shape) != tuple(policy_logprobs.shape):
            raise RuntimeError("response_mask must align with policy response logprobs.")
        if int(rewards.numel()) != int(policy_logprobs.size(0)):
            raise RuntimeError("rewards must contain exactly one value per rollout.")

        # Actor trajectories are normally finite, but never let malformed
        # optional old-policy values turn a smoke test into ``NaN * 0``.
        # The objective itself only uses the fresh gradient-bearing policy
        # signal; Actor pi_old remains part of this plugin's requested schema
        # so the real Actor -> Learner collation path is exercised too.
        finite_policy = torch.isfinite(policy_logprobs)
        valid_mask = response_mask.to(dtype=torch.bool) & finite_policy
        safe_policy = torch.where(valid_mask, policy_logprobs, torch.zeros_like(policy_logprobs))
        token_counts = valid_mask.sum(dim=1).to(dtype=policy_logprobs.dtype).clamp_min(1.0)
        rollout_logprobs = safe_policy.sum(dim=1) / token_counts
        signed_rewards = rewards.mul(2.0).sub(1.0)
        objective = (rollout_logprobs * signed_rewards).mean()
        no_grad_mean = masked_token_mean(
            no_grad_logprobs,
            response_mask,
            finite_only=True,
            shape_error=_SMOKE_METRIC_MASK_SHAPE_ERROR,
        )
        reference_mean = masked_token_mean(
            reference_logprobs,
            response_mask,
            finite_only=True,
            shape_error=_SMOKE_METRIC_MASK_SHAPE_ERROR,
        )
        alternate_mean = masked_token_mean(
            alternate_logprobs,
            response_mask,
            finite_only=True,
            shape_error=_SMOKE_METRIC_MASK_SHAPE_ERROR,
        )
        prompt_id_mean = masked_token_mean(
            prompt_id_logprobs,
            response_mask,
            finite_only=True,
            shape_error=_SMOKE_METRIC_MASK_SHAPE_ERROR,
        )
        probe_probs = require_tensor_batch_field(batch, "answer_probe_probs", dtype=torch.float32)
        actor_lora_step = _actor_lora_step(batch)

        return objective, {
            "smoke_reward_mean": rewards.detach().float().mean(),
            "smoke_loss": -objective.detach(),
            "smoke_policy_no_grad_logprob": no_grad_mean.detach(),
            "smoke_reference_logprob": reference_mean.detach(),
            "smoke_reference_policy_gap": (no_grad_mean - reference_mean).abs().detach(),
            "smoke_alternate_prompt_logprob": alternate_mean.detach(),
            "smoke_prompt_id_logprob": prompt_id_mean.detach(),
            "smoke_prompt_id_gap": (prompt_id_mean - no_grad_mean).abs().detach(),
            "smoke_entropy": masked_token_mean(
                entropy,
                response_mask,
                finite_only=True,
                shape_error=_SMOKE_METRIC_MASK_SHAPE_ERROR,
            ).detach(),
            "smoke_topk_width": float(topk_ids.size(-1)),
            "smoke_probe_used_fraction": (probe_probs[:, 0] >= 0).float().mean(),
            # These make the subprocess integration smoke observable without
            # exposing a Learner runtime object to algorithm code.
            "smoke_reference_calls": 1.0,
            "smoke_policy_no_grad_calls": 3.0,
            "smoke_policy_grad_calls": 1.0,
            "smoke_prompt_override_calls": 1.0,
            "smoke_prompt_id_override_calls": 1.0,
            "smoke_policy_grad_has_graph": 1.0,
            "smoke_actor_lora_step": float(actor_lora_step),
        }

    def _render_alternate_prompt(self) -> str:
        """Create one no-thinking prompt used solely to exercise overrides."""

        return apply_generation_chat_template(
            self._tokenizer,
            [
                {
                    "role": "user",
                    "content": "Read the following response context carefully and continue the solution.",
                }
            ],
            enable_thinking=self._enable_thinking,
        )


def _actor_lora_step(batch: Mapping[str, object]) -> int:
    """Expose the Actor's detached adapter version as a smoke metric only."""

    actor_metrics = batch.get("actor_metrics")
    if not isinstance(actor_metrics, Mapping):
        return -1
    value = actor_metrics.get("lora_step", -1)
    if isinstance(value, bool):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


__all__ = ["ActorTestPlugin", "ChatTemplateTokenizer"]
