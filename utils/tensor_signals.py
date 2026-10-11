"""Algorithm-neutral helpers for tensor-valued signals and batch fields.

Algorithm plugins should express which model signals they need and how those
signals form an objective.  The repetitive mechanics of validating a
tensor-valued signal, reading a collated batch field, and reducing masked
token values belong here instead.  These helpers intentionally never move a
tensor between devices, except for :func:`require_cpu_vector`, whose explicit
purpose is to produce a detached CPU-side vector for control-flow-only work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def require_tensor_signal(signals: Mapping[str, object], key: str) -> torch.Tensor:
    """Return one tensor emitted by ``compute_logits()``.

    This is deliberately a narrow contract: algorithm code decides the signal
    name and any shape semantics, while the common error for a missing or
    non-tensor model signal remains consistent across plugins.
    """

    value = signals.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"compute_logits() did not return tensor signal {key!r}.")
    return value


def require_tensor_batch_field(
    batch: Mapping[str, object],
    key: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Read one collated tensor field without moving it off its current device."""

    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor.")
    return value.to(dtype=dtype) if dtype is not None else value


def require_tensor_sequence(
    signals: Mapping[str, object],
    key: str,
    *,
    producer: str = "compute_logits()",
    description: str | None = None,
) -> list[torch.Tensor]:
    """Return a sequence containing tensors only, preserving its item order.

    Ragged Teacher outputs are one current use case.  Their rollout alignment
    and per-row shape remain algorithm-owned checks; this utility only owns
    the generic tensor-sequence boundary.
    """

    value = signals.get(key)
    expected_description = description or f"a tensor sequence for {key!r}"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{producer} must return {expected_description}.")
    values = list(value)
    if not all(isinstance(item, torch.Tensor) for item in values):
        raise TypeError(f"{producer} {key} must contain only tensors.")
    return values


def require_cpu_vector(
    value: object,
    name: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a detached rank-one tensor on CPU in ``dtype``.

    Use this only for non-differentiable control-flow work such as grouping
    rollout rewards.  Objective code should keep using the original
    device-local tensors through :func:`require_tensor_batch_field`.
    """

    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a rank-one torch.Tensor.")
    return value.detach().cpu().to(dtype=dtype)


def masked_token_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    finite_only: bool = False,
    shape_error: str = "Token values and masks must have equal shapes.",
) -> torch.Tensor:
    """Compute a safe masked token mean while preserving a gradient graph.

    ``mask`` identifies valid token positions.  With ``finite_only=True``,
    non-finite values are excluded as well; callers that already construct a
    finite validity mask can leave it at the default.  ``shape_error`` lets a
    public plugin contract preserve its existing validation message.  An
    empty valid mask yields zero instead of a NaN.
    """

    if tuple(values.shape) != tuple(mask.shape):
        raise RuntimeError(shape_error)
    valid = mask.to(dtype=torch.bool)
    if finite_only:
        valid = valid & torch.isfinite(values)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    return safe_values.sum() / valid.sum().to(dtype=values.dtype).clamp_min(1.0)


def sequence_masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average valid tokens per sequence, then average valid sequences equally."""

    if tuple(values.shape) != tuple(mask.shape):
        raise RuntimeError("Token values and masks must have equal shapes.")
    if values.ndim != 2:
        raise RuntimeError("Sequence token values and masks must be rank-two.")
    valid = mask.to(dtype=torch.bool)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    token_counts = valid.sum(dim=1).to(dtype=values.dtype).clamp_min(1.0)
    sequence_means = safe_values.sum(dim=1) / token_counts
    valid_sequences = valid.any(dim=1)
    return (sequence_means * valid_sequences.to(dtype=values.dtype)).sum() / valid_sequences.sum().to(
        dtype=values.dtype
    ).clamp_min(1.0)


__all__ = [
    "masked_token_mean",
    "require_cpu_vector",
    "require_tensor_batch_field",
    "require_tensor_sequence",
    "require_tensor_signal",
    "sequence_masked_token_mean",
]
