"""Outcome-conditioned self-distillation with the current LoRA student."""

from __future__ import annotations

import copy

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig, apply_generation_chat_template, controller_questions_and_answers,
    require_chat_template_tokenizer, require_train_template, require_training_sampling,
)
from utils.answer_verifier import verify_response_answers
from utils.ocsd import validate_ocsd_parameters
from utils.prompts import render_prompt
from utils.tensor_signals import masked_token_mean, require_tensor_batch_field

from .algorithm_base import AlgorithmPlugin, ObjectiveResult, Signals
from .plugin_registry import register_plugin_class


@register_plugin_class
class OCSDPlugin(AlgorithmPlugin):
    def __init__(self, *, tokenizer: object, runtime: AlgorithmRuntimeConfig) -> None:
        self._tokenizer = require_chat_template_tokenizer(tokenizer)
        self._train_template = require_train_template(runtime)
        self._sampling = require_training_sampling(runtime)
        self._enable_thinking = runtime.enable_thinking
        if runtime.algorithm.get("loss_aggregation") != "token":
            raise ValueError("OCSD requires algorithm.loss_aggregation='token'.")
        if self._sampling.get("temperature", 1.0) != 1.0:
            raise ValueError("OCSD requires algorithm.training_sampling.temperature=1.0.")
        self._gamma, self._tau = validate_ocsd_parameters(
            runtime.algorithm.get("ocsd_gamma", 0.1), runtime.algorithm.get("ocsd_tau", 0.95),
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
            },
        }

    def collect_learner_signals(self, signals: Signals) -> Signals:
        # Learner has already validated that prompt indices align with responses
        # and point into the logical batch's answer list.
        responses = signals["responses"]
        answers = signals["answers"]
        rollout_answers = [str(answers[index]) for index in signals["prompt_indices"].tolist()]
        scores = verify_response_answers(responses, rollout_answers)
        return {"ocsd_correct": torch.tensor(scores, dtype=torch.bool)}

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        values = self.compute_ocsd(batch, gamma=self._gamma, tau=self._tau)
        mask = require_tensor_batch_field(batch, "response_mask", dtype=torch.bool)
        correct = require_tensor_batch_field(batch, "ocsd_correct")
        loss = masked_token_mean(values["response_ocsd_kl"], mask)
        transfer = values["response_ocsd_transfer"]
        return ObjectiveResult(
            objective=-loss,
            metrics={
                "ocsd_loss": loss.detach(),
                "ocsd_accuracy": correct.float().mean(),
                "ocsd_head_mass": masked_token_mean(values["response_ocsd_head_mass"], mask),
                "ocsd_transfer_mass": masked_token_mean(transfer, mask),
                "ocsd_zero_transfer_fraction": masked_token_mean((transfer == 0).float(), mask),
            },
            valid_token_count=mask.sum().detach(),
        )


__all__ = ["OCSDPlugin"]
