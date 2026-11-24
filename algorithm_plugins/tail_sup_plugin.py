"""On-policy Student-top-k tail suppression, without rewards or a teacher."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig, apply_generation_chat_template, controller_questions_and_answers,
    require_chat_template_tokenizer, require_train_template, require_training_sampling,
)
from utils.prompts import render_prompt
from utils.tail_suppression import validate_tail_sup_parameters
from utils.tensor_signals import masked_token_mean
from utils.topk_kl import validate_kl_top_k

from .algorithm_base import AlgorithmPlugin, ObjectiveResult, Signals
from .plugin_registry import register_plugin_class


@register_plugin_class
class TailSupPlugin(AlgorithmPlugin):
    """Distil each rollout policy's renormalized top-p head into the student."""

    requires_frozen_reference = True

    def __init__(self, *, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        if runtime.algorithm.get("loss_aggregation") != "token":
            raise ValueError("tail_sup requires algorithm.loss_aggregation='token'.")
        interval = runtime.algorithm.get("teacher_sync_interval")
        if type(interval) is not int or interval != 1:
            raise ValueError("tail_sup requires teacher_sync_interval=1 for versioned old-student snapshots.")
        self._top_p, self._temperature = validate_tail_sup_parameters(
            runtime.algorithm.get("tail_sup_top_p", 0.99),
            runtime.algorithm.get("tail_sup_temperature", 1.0),
        )
        self._kl_top_k = validate_kl_top_k(runtime.algorithm.get("kl_top_k", 512))

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> "TailSupPlugin":
        return cls(tokenizer=tokenizer, runtime=runtime)

    def collect_actor_signals(self, signals: Signals) -> Signals:
        questions, _answers = controller_questions_and_answers(signals)
        prompts = [apply_generation_chat_template(
            self._tokenizer,
            [{"role": "user", "content": render_prompt(self._train_template, question=question)}],
            enable_thinking=self._enable_thinking,
        ) for question in questions]
        return {
            "prompts": prompts,
            "sampling": copy.deepcopy(self._sampling),
            "return_signals": {"prompt_ids": True, "response_ids": True, "prompt_indices": True,
                               "responses": True, "response_truncated": True},
        }

    def collect_learner_signals(self, signals: Signals) -> Signals:
        # No correctness labels, rewards, solution prompts or full-vocabulary
        # CPU caches. The Learner scores the exact rollout adapter locally.
        version = signals.get("policy_version")
        if type(version) is not int or version < 0:
            raise ValueError("tail_sup requires the rollout's non-negative integer policy_version.")
        if signals.get("behavior_version", version) != version:
            raise ValueError("tail_sup rollout behavior_version and policy_version disagree.")
        return {}

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        values = self.compute_tail_suppression(
            batch, top_p=self._top_p, temperature=self._temperature,
        )
        response_mask = batch.get("response_mask")
        if not isinstance(response_mask, torch.Tensor):
            raise TypeError("tail_sup requires a tensor response_mask.")
        mask = response_mask.bool() & values["response_tail_sup_active"]
        loss = masked_token_mean(values["response_tail_sup_kl"], mask)
        metrics = {"tail_sup_loss": loss.detach(), "tail_sup_top_p": self._top_p}
        for name in ("head_mass", "head_size", "old_tail_mass", "tail_mass", "entropy", "max_probability"):
            value = values[f"response_tail_sup_{name}"].to(dtype=loss.dtype)
            metrics[f"tail_sup_{name}"] = masked_token_mean(value, mask).detach()
        metrics["tail_sup_skip_fraction"] = masked_token_mean(
            (~values["response_tail_sup_active"]).to(dtype=loss.dtype), response_mask.bool(),
        ).detach()
        return ObjectiveResult(objective=-loss, metrics=metrics, valid_token_count=mask.sum().detach())

    def training_state_dict(self) -> dict[str, object]:
        return {"version": 1, "algorithm": "tail_sup", "top_p": self._top_p,
                "temperature": self._temperature, "loss_aggregation": "token",
                "target_policy": "rollout_policy_version", "skip_rule": "max_probability > top_p",
                "distribution_support": "student_top_k", "kl_top_k": self._kl_top_k}

    def load_training_state_dict(self, state: Mapping[str, object], *, step: int) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("tail_sup checkpoint state must be a mapping.")
        if not state and step == 0:
            return
        expected = self.training_state_dict()
        if set(state) != set(expected) or any(
            type(state[key]) is not type(value) or state[key] != value for key, value in expected.items()
        ):
            raise ValueError("tail_sup checkpoint objective settings do not match this run.")


__all__ = ["TailSupPlugin"]
