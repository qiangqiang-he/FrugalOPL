"""Non-interpolating empirical-quantile baselines for group advantages."""

from __future__ import annotations

import math

import torch


def validate_qae_quantile(value: object) -> float:
    """Require an interior empirical-CDF quantile, excluding booleans."""

    message = "algorithm.qae_quantile must be a finite number strictly between 0 and 1."
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        quantile = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError(message)
    return quantile


def empirical_quantile_baseline(rewards: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return inverse empirical CDF ``r_(ceil(quantile * G))``.

    The baseline is an observed reward, never a linear interpolation between
    adjacent order statistics. For binary groups of eight and quantile 0.4,
    it is zero with up to four successes and one with five or more successes.
    """

    quantile = validate_qae_quantile(quantile)
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("QAE rewards must be a non-empty one-dimensional tensor.")
    if rewards.dtype == torch.bool or rewards.is_complex() or not bool(torch.isfinite(rewards).all()):
        raise ValueError("QAE rewards must contain finite real numeric values.")
    rank = math.ceil(quantile * rewards.numel())
    return rewards.kthvalue(rank).values


__all__ = ["empirical_quantile_baseline", "validate_qae_quantile"]
