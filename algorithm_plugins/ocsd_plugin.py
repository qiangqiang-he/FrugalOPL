"""Outcome-conditioned self-distillation with the current LoRA student."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig, apply_generation_chat_template, controller_questions_and_answers,
    require_chat_template_tokenizer, require_train_template, require_training_sampling,
)
from utils.answer_verifier import verify_response_answers
from utils.ocsd import validate_ocsd_parameters
from utils.prompts import render_prompt
from utils.tensor_signals import masked_token_mean, require_tensor_batch_field
from utils.topk_kl import validate_kl_top_k

from .algorithm_base import AlgorithmPlugin, ObjectiveResult, Signals
from .plugin_registry import register_plugin_class


@register_plugin_class
class OCSDPlugin(AlgorithmPlugin):
    def __init__(self, *, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        self._kl_top_k = validate_kl_top_k(runtime.algorithm.get("kl_top_k", 512))
        if runtime.algorithm.get("loss_aggregation") != "token":
            raise ValueError("OCSD requires algorithm.loss_aggregation='token'.")
        if self._sampling.get("temperature", 1.0) != 1.0:
            raise ValueError("OCSD requires algorithm.training_sampling.temperature=1.0.")
        self._gamma_pos, self._tau = validate_ocsd_parameters(
            runtime.algorithm.get("ocsd_gamma", 0.25), runtime.algorithm.get("ocsd_tau", 0.95),
        )

    @classmethod
    def from_runtime_config(cls, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> OCSDPlugin:
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
            "return_signals": {
                "prompt_ids": True, "response_ids": True, "prompt_indices": True, "responses": True,
                "response_logprobs": False, "response_truncated": False,
                "generation_entropy": True,
            },
        }

    def collect_learner_signals(self, signals: Signals) -> Signals:
        # Learner has already validated that prompt indices align with responses
        # and point into the logical batch's answer list.
        responses = signals["responses"]
        answers = signals["answers"]
        rollout_answers = [str(answers[index]) for index in signals["prompt_indices"].tolist()]
        scores = verify_response_answers(responses, rollout_answers)
        correct_count = sum(scores)
        incorrect_count = len(scores) - correct_count
        # This hook sees the complete logical batch before it is split across ranks.
        negative_scale = min(0.75, max(0.25, correct_count / max(1, incorrect_count)))
        return {
            "ocsd_correct": torch.tensor(scores, dtype=torch.bool),
            "ocsd_gamma_neg": self._gamma_pos * negative_scale,
        }

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        gamma_neg = float(batch["ocsd_gamma_neg"])
        values = self.compute_ocsd(batch, gamma_pos=self._gamma_pos, gamma_neg=gamma_neg, tau=self._tau)
        mask = require_tensor_batch_field(batch, "response_mask", dtype=torch.bool)
        correct = require_tensor_batch_field(batch, "ocsd_correct")
        loss = masked_token_mean(values["response_ocsd_kl"], mask)
        reward = correct.float().mean()
        outcome = "correct" if bool(correct.item()) else "incorrect"
        metrics = {
            "ocsd_loss": loss.detach(),
            "ocsd_accuracy": reward,
            "ocsd_reward_mean": reward,
            "ocsd_response_length_mean": mask.sum(-1).float().mean(),
            "ocsd_gamma_pos": self._gamma_pos,
            "ocsd_gamma_neg": gamma_neg,
            "ocsd_zero_transfer_fraction": masked_token_mean((values["response_ocsd_transfer"] == 0).float(), mask),
        }
        for name in ("head_mass", "tail_mass", "protected_mass", "transfer",
                     "head_relative_transfer", "tail_relative_transfer"):
            mean = masked_token_mean(values[f"response_ocsd_{name}"], mask)
            metric_name = "transfer_mass" if name == "transfer" else name
            metrics[f"ocsd_{metric_name}"] = mean
            metrics[f"ocsd_{outcome}_{metric_name}"] = mean
        return ObjectiveResult(
            objective=-loss,
            metrics=metrics,
            valid_token_count=mask.sum().detach(),
        )

    def training_state_dict(self):
        return {"version": 1, "algorithm": "OCSD", "distribution_support": "student_top_k",
                "kl_top_k": self._kl_top_k, "gamma_pos": self._gamma_pos, "tau": self._tau}

    def load_training_state_dict(self, state, *, step):
        if not state and step == 0:
            return
        expected = self.training_state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected) or any(
            type(state[key]) is not type(value) or state[key] != value for key, value in expected.items()
        ):
            raise ValueError("OCSD checkpoint objective settings do not match this run.")


__all__ = ["OCSDPlugin"]
