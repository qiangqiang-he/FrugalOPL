"""Entropy-window control for GRPO's negative advantages.

Only completed logical training steps enter this state machine.  In
particular, evaluations and Actor rollout waves never advance its counters.
The returned coefficient always applies to the *next* optimizer step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


_STATE_VERSION = 3


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite non-negative number.") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


@dataclass(frozen=True)
class DynamicAdvantageConfig:
    enabled: bool = False
    baseline_steps: int = 100
    window_steps: int = 10
    min_negative_scale: float = 0.1

    @classmethod
    def from_mapping(cls, raw: object) -> "DynamicAdvantageConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm.dynamic_advantage must be a mapping.")
        allowed = {"enabled", "baseline_steps", "window_steps", "min_negative_scale"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unsupported algorithm.dynamic_advantage settings: {sorted(unknown)}.")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("algorithm.dynamic_advantage.enabled must be a boolean.")
        baseline = _integer(raw.get("baseline_steps", 100), "dynamic_advantage.baseline_steps", minimum=1)
        window = _integer(raw.get("window_steps", 10), "dynamic_advantage.window_steps", minimum=1)
        floor = _number(raw.get("min_negative_scale", 0.1), "dynamic_advantage.min_negative_scale")
        if not 0.1 <= floor <= 1.0:
            raise ValueError("dynamic_advantage.min_negative_scale must be in [0.1, 1.0].")
        return cls(enabled=enabled, baseline_steps=baseline, window_steps=window, min_negative_scale=floor)

    def state_config(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "baseline_steps": self.baseline_steps,
            "window_steps": self.window_steps,
            "min_negative_scale": self.min_negative_scale,
        }


class DynamicAdvantage:
    """Keep a fixed initial entropy baseline and non-overlapping windows."""

    def __init__(self, config: DynamicAdvantageConfig) -> None:
        self.config = config
        self._reset()

    def _reset(self) -> None:
        self.completed_steps = 0
        self.baseline_count = 0
        self.baseline_sum = 0.0
        self.baseline_values: list[float] = []
        self.baseline_entropy: float | None = None
        self.window_count = 0
        self.window_sum = 0.0
        self.window_values: list[float] = []
        self.training_entropy: float | None = None
        self.entropy_ratio: float | None = None
        self.alpha_neg = 1.0

    def coefficient(self, ratio: float) -> float:
        ratio = _number(ratio, "dynamic advantage entropy ratio")
        # Closed upper edges keep exact threshold values in the lower band.
        # Direct comparisons also handle arbitrarily large finite ratios
        # without arithmetic rounding or overflow at a band boundary.
        for upper, scale in (
            (1.5, 1.0),
            (2.0, 0.9),
            (3.0, 0.8),
            (4.0, 0.7),
            (5.0, 0.6),
            (6.0, 0.5),
            (7.0, 0.4),
            (8.0, 0.3),
            (9.0, 0.2),
        ):
            if ratio <= upper:
                return max(self.config.min_negative_scale, scale)
        return self.config.min_negative_scale

    def observe(self, *, step: int, entropy: object) -> dict[str, float]:
        if not self.config.enabled:
            return {}
        step = _integer(step, "dynamic advantage completed step", minimum=1)
        if step != self.completed_steps + 1:
            raise ValueError(
                f"Dynamic advantage requires consecutive completed steps; "
                f"expected {self.completed_steps + 1}, got {step}."
            )
        value = _number(entropy, "grpo_entropy")
        applied_alpha = self.alpha_neg
        baseline_count, baseline_sum = self.baseline_count, self.baseline_sum
        baseline_values = self.baseline_values
        baseline_entropy = self.baseline_entropy
        window_count, window_sum = self.window_count, self.window_sum
        window_values = self.window_values
        training_entropy, entropy_ratio, next_alpha = self.training_entropy, self.entropy_ratio, self.alpha_neg

        if step <= self.config.baseline_steps:
            baseline_count += 1
            baseline_values = [*baseline_values, value]
            baseline_sum = _number(math.fsum(baseline_values), "dynamic advantage baseline sum")
            if baseline_count == self.config.baseline_steps:
                baseline_entropy = baseline_sum / baseline_count
                if baseline_entropy <= 0:
                    raise ValueError("Dynamic advantage baseline entropy must be positive after warmup.")
        else:
            if baseline_entropy is None or baseline_entropy <= 0:
                raise RuntimeError("Dynamic advantage has no valid frozen baseline entropy.")
            window_count += 1
            window_values = [*window_values, value]
            window_sum = _number(math.fsum(window_values), "dynamic advantage window sum")
            if window_count == self.config.window_steps:
                training_entropy = window_sum / window_count
                entropy_ratio = _number(training_entropy / baseline_entropy, "dynamic advantage entropy ratio")
                next_alpha = self.coefficient(entropy_ratio)
                window_count, window_sum = 0, 0.0
                window_values = []

        # Commit only after all validation succeeds, so invalid metrics cannot
        # partially advance a baseline or window.
        self.completed_steps = step
        self.baseline_count, self.baseline_sum = baseline_count, baseline_sum
        self.baseline_values = baseline_values
        self.baseline_entropy = baseline_entropy
        self.window_count, self.window_sum = window_count, window_sum
        self.window_values = window_values
        self.training_entropy, self.entropy_ratio, self.alpha_neg = training_entropy, entropy_ratio, next_alpha

        diagnostics = {
            "dynamic_advantage_enabled": 1.0,
            "dynamic_advantage_alpha_neg": applied_alpha,
            "dynamic_advantage_alpha_neg_next": next_alpha,
            "dynamic_advantage_baseline_entropy": baseline_sum / baseline_count,
            "dynamic_advantage_baseline_count": float(baseline_count),
            "dynamic_advantage_baseline_ready": float(baseline_entropy is not None),
            "dynamic_advantage_window_count": float(window_count),
        }
        if window_count:
            diagnostics["dynamic_advantage_window_running_entropy"] = window_sum / window_count
        if training_entropy is not None:
            diagnostics["dynamic_advantage_training_entropy"] = training_entropy
        if entropy_ratio is not None:
            diagnostics["dynamic_advantage_entropy_ratio"] = entropy_ratio
        return diagnostics

    def state_dict(self) -> dict[str, object]:
        if not self.config.enabled:
            return {}
        return {
            "version": _STATE_VERSION,
            "config": self.config.state_config(),
            "completed_steps": self.completed_steps,
            "baseline_count": self.baseline_count,
            "baseline_sum": self.baseline_sum,
            "baseline_values": list(self.baseline_values),
            "baseline_entropy": self.baseline_entropy,
            "window_count": self.window_count,
            "window_sum": self.window_sum,
            "window_values": list(self.window_values),
            "training_entropy": self.training_entropy,
            "entropy_ratio": self.entropy_ratio,
            "alpha_neg": self.alpha_neg,
        }

    def load_state_dict(self, state: Mapping[str, object], *, step: int) -> None:
        step = _integer(step, "dynamic advantage checkpoint step")
        if not isinstance(state, Mapping):
            raise ValueError("Dynamic advantage checkpoint state must be a mapping.")
        if not self.config.enabled:
            if state:
                raise ValueError("Cannot restore dynamic advantage state while the strategy is disabled.")
            return
        if not state:
            if step:
                raise ValueError("Cannot resume dynamic advantage without its saved training state.")
            self._reset()
            return
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError("Dynamic advantage checkpoint has missing or unsupported state fields.")
        if type(state["version"]) is not int or state["version"] != _STATE_VERSION:
            raise ValueError(
                f"Unsupported dynamic advantage checkpoint version; expected {_STATE_VERSION}. "
                "A checkpoint from the previous entropy thresholds cannot resume this policy."
            )
        restored_config = DynamicAdvantageConfig.from_mapping(state["config"])
        if restored_config != self.config:
            raise ValueError("Dynamic advantage checkpoint configuration does not match this run.")
        completed = _integer(state["completed_steps"], "dynamic advantage saved step")
        if completed != step:
            raise ValueError("Dynamic advantage checkpoint step does not match the restored optimizer.")
        count = _integer(state["baseline_count"], "dynamic advantage saved baseline count")
        total = _number(state["baseline_sum"], "dynamic advantage saved baseline sum")
        baseline_values = self._saved_values(state["baseline_values"], "baseline")
        if count != min(step, self.config.baseline_steps) or (count == 0 and total != 0):
            raise ValueError("Dynamic advantage checkpoint has an inconsistent baseline count or sum.")
        if len(baseline_values) != count or math.fsum(baseline_values) != total:
            raise ValueError("Dynamic advantage checkpoint baseline samples do not match their count or sum.")
        baseline = state["baseline_entropy"]
        if step < self.config.baseline_steps:
            if baseline is not None:
                raise ValueError("Dynamic advantage baseline was frozen before warmup completed.")
        else:
            baseline = _number(baseline, "dynamic advantage saved baseline entropy")
            if baseline <= 0 or baseline != total / count:
                raise ValueError("Dynamic advantage checkpoint baseline entropy is inconsistent.")
        window_count = _integer(state["window_count"], "dynamic advantage saved window count")
        window_sum = _number(state["window_sum"], "dynamic advantage saved window sum")
        window_values = self._saved_values(state["window_values"], "window")
        if window_count != max(0, step - self.config.baseline_steps) % self.config.window_steps:
            raise ValueError("Dynamic advantage checkpoint has an inconsistent partial window count.")
        if window_count == 0 and window_sum != 0:
            raise ValueError("Dynamic advantage checkpoint has a sum for an empty partial window.")
        if len(window_values) != window_count or math.fsum(window_values) != window_sum:
            raise ValueError("Dynamic advantage checkpoint window samples do not match their count or sum.")
        training, ratio = state["training_entropy"], state["entropy_ratio"]
        alpha = _number(state["alpha_neg"], "dynamic advantage saved coefficient")
        if step < self.config.baseline_steps + self.config.window_steps:
            if training is not None or ratio is not None or alpha != 1.0:
                raise ValueError("Dynamic advantage coefficient changed before the first complete entropy window.")
        else:
            training = _number(training, "dynamic advantage saved training entropy")
            ratio = _number(ratio, "dynamic advantage saved entropy ratio")
            if baseline is None or ratio != training / baseline:
                raise ValueError("Dynamic advantage checkpoint entropy ratio is inconsistent.")
            if alpha != self.coefficient(ratio):
                raise ValueError("Dynamic advantage checkpoint coefficient is inconsistent with its entropy ratio.")
        self.completed_steps = step
        self.baseline_count, self.baseline_sum, self.baseline_entropy = count, total, baseline
        self.baseline_values = baseline_values
        self.window_count, self.window_sum = window_count, window_sum
        self.window_values = window_values
        self.training_entropy, self.entropy_ratio, self.alpha_neg = training, ratio, alpha

    @staticmethod
    def _saved_values(raw: object, name: str) -> list[float]:
        if not isinstance(raw, list):
            raise ValueError(f"Dynamic advantage saved {name} entropy samples must be a list.")
        return [_number(value, f"dynamic advantage saved {name} entropy") for value in raw]


__all__ = ["DynamicAdvantage", "DynamicAdvantageConfig"]
