"""TCDO: zero transfer on success, p²/q on error, Teacher KL on truncation."""
from __future__ import annotations

import math
import torch
from utils.answer_verifier import verify_response_answers
from utils.tcdo import CORRECT, INCORRECT, TRUNCATED
from utils.tensor_signals import masked_token_mean
from .algorithm_base import ObjectiveResult, Signals
from .plugin_registry import register_plugin_class
from .self_shot_opsd_plugin import SelfShotOPSDPlugin


@register_plugin_class
class TCDOPlugin(SelfShotOPSDPlugin):
    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        self.fraction = self._parameter(runtime, "tcdo_plausible_fraction", 0.01, positive=True)
        if self.fraction > 1:
            raise ValueError("tcdo_plausible_fraction must lie in (0,1]")
        self.incorrect_weight = self._parameter(runtime, "tcdo_incorrect_weight", 1.0)
        self.truncated_weight = self._parameter(runtime, "tcdo_truncated_weight", 1.0)
        if self._kl_token_clip is not None or runtime.algorithm.get("teacher_sync_interval", -1) != -1:
            raise ValueError("TCDO requires kl_token_clip=null and a frozen base Teacher.")

    @staticmethod
    def _parameter(runtime, name, default, positive=False):
        value = runtime.algorithm.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {name}")
        return float(value)

    def collect_actor_signals(self, signals: Signals) -> Signals:
        request = super().collect_actor_signals(signals)
        request["return_signals"] = {**request["return_signals"], "responses": True, "response_truncated": True}
        return request

    def collect_learner_signals(self, signals: Signals) -> Signals:
        additions = super().collect_learner_signals(signals)
        indices = signals["prompt_indices"].tolist()
        truncated = signals["response_truncated"].tolist()
        responses = signals["responses"]
        state = [TRUNCATED if cut else INCORRECT for cut in truncated]
        active = [i for i, cut in enumerate(truncated) if not cut]
        scores = verify_response_answers([responses[i] for i in active], [str(signals["answers"][indices[i]]) for i in active])
        for i, score in zip(active, scores, strict=True):
            if score == 1:
                state[i] = CORRECT
        lengths = [int(row.numel()) for row in signals["response_ids"]]
        count, tokens = len(state), sum(lengths)
        stats = {}
        for branch, label in [(CORRECT, "correct"), (INCORRECT, "incorrect"), (TRUNCATED, "truncated")]:
            stats[label + "_fraction"] = sum(s == branch for s in state) / count
            stats[label + "_token_fraction"] = sum(n for n, s in zip(lengths, state) if s == branch) / tokens
        stats.update(incorrect_weight=self.incorrect_weight, truncated_weight=self.truncated_weight,
                     plausible_fraction=self.fraction)
        return {**additions, "tcdo_state": torch.tensor(state, dtype=torch.int64), "tcdo_batch_stats": stats}

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        state = int(batch["tcdo_state"].item())
        mask = batch["response_mask"].bool()
        if state == CORRECT:
            values = self.compute_logits(batch, mode="policy_no_grad", return_signals={
                "response_entropy": True, "entropy_requires_grad": False})
            entropy = masked_token_mean(values["response_entropy"], mask).detach()
            loss = entropy.new_zeros((), requires_grad=True)
            metrics = {"tcdo_kl_loss": loss.detach(), "student_entropy_nats": entropy, "target_entropy_nats": entropy}
            weight = 0.0
        else:
            values = self.compute_tcdo(batch, teacher_prompts=batch["teacher_prompts"], state=state,
                                       temperature=self._kl_temperature, plausible_fraction=self.fraction)
            weight = self.incorrect_weight if state == INCORRECT else self.truncated_weight
            loss = masked_token_mean(values["tcdo_kl_loss"], mask) * weight
            metrics = {name: masked_token_mean(value, mask).detach() for name, value in values.items()}
            metrics["tcdo_kl_loss"] = loss.detach()
        token_metrics = frozenset(metrics)
        metrics.update(batch["tcdo_batch_stats"])
        return ObjectiveResult(-loss, metrics, mask.sum().detach(), token_metrics, weight > 0)

    def training_state_dict(self):
        return {**super().training_state_dict(), "algorithm": "tcdo", "plausible_fraction": self.fraction,
                "incorrect_weight": self.incorrect_weight, "truncated_weight": self.truncated_weight}
