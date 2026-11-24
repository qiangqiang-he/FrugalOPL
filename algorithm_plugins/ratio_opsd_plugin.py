"""Fit detached p²/q targets using the frozen self-shot privileged Teacher."""

from __future__ import annotations

import math

import torch

from utils.tensor_signals import masked_token_mean

from .algorithm_base import ObjectiveResult, Signals
from .plugin_registry import register_plugin_class
from .self_shot_opsd_plugin import SelfShotOPSDPlugin


@register_plugin_class
class RatioOPSDPlugin(SelfShotOPSDPlugin):
    """Student-top-k KL with no CLIP, reward or prefix weight."""

    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        self._plausible_fraction = runtime.algorithm.get("ratio_plausible_fraction", 0.1)
        if (isinstance(self._plausible_fraction, bool)
                or not isinstance(self._plausible_fraction, (int, float))
                or not math.isfinite(self._plausible_fraction)
                or not 0 < self._plausible_fraction <= 1):
            raise ValueError("ratio_plausible_fraction must lie in (0, 1].")
        self._plausible_fraction = float(self._plausible_fraction)
        if runtime.algorithm.get("teacher_sync_interval", -1) != -1:
            raise ValueError("Ratio-OPSD requires the frozen base Teacher: teacher_sync_interval=-1.")
        if self._kl_token_clip is not None:
            raise ValueError("Ratio-OPSD requires kl_token_clip: null; its target has no CLIP.")

    def collect_actor_signals(self, signals: Signals) -> Signals:
        request = super().collect_actor_signals(signals)
        request["return_signals"] = {**request["return_signals"], "response_truncated": True}
        return request

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        values = self.compute_ratio_kl(
            batch, teacher_prompts=batch.get("teacher_prompts"),
            temperature=self._kl_temperature, plausible_fraction=self._plausible_fraction,
        )
        mask = batch["response_mask"].bool()
        loss = masked_token_mean(values["ratio_kl_loss"], mask)
        metrics = {name: masked_token_mean(value, mask).detach() for name, value in values.items()}
        return ObjectiveResult(-loss, metrics, mask.sum().detach())

    def training_state_dict(self):
        return {**super().training_state_dict(), "algorithm": "ratio_opsd", "target_power": 2,
                "plausible_fraction": self._plausible_fraction}
