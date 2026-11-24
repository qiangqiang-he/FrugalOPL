"""Student-top-k OPSD Forward KL with the pointwise upper clip."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from utils.topk_kl import DEFAULT_KL_TOP_K, student_topk_logits


def forward_kl_token_values(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_clip: float | None = None,
    top_k: int = DEFAULT_KL_TOP_K,
) -> dict[str, torch.Tensor]:
    """Return response-position values of ``KL(Teacher || Student)``.

    Upstream's ``jsd_token_clip`` caps each vocabulary contribution *before*
    summing vocabulary. This is not a cap on the final per-token KL, and its
    clipped sum can be negative. There is no temperature-squared multiplier.
    Teacher values and logging diagnostics never receive a gradient.

    Normalize only the selected support in FP32 (FP64 for FP64 inputs).
    Process the whole response, checkpointing the head/loss once.
    """

    if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
        raise TypeError("Forward KL logits must be tensors.")
    if student_logits.ndim < 2 or student_logits.shape != teacher_logits.shape or student_logits.size(-1) < 1:
        raise ValueError("Forward KL logits must have equal shapes with a non-empty vocabulary dimension.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("Forward KL logits must have floating-point dtypes.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Forward KL logits must be on the same device.")
    for name, value in (("temperature", temperature), ("token_clip", token_clip)):
        if value is None and name == "token_clip":
            continue
        if isinstance(value, bool):
            raise ValueError(f"Forward KL {name} must be a positive finite number.")
        try:
            valid = math.isfinite(float(value)) and float(value) > 0
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Forward KL {name} must be a positive finite number.") from error
        if not valid:
            raise ValueError(f"Forward KL {name} must be a positive finite number.")

    student_logits, teacher_logits = student_topk_logits(student_logits, teacher_logits, top_k=top_k)
    dtype = torch.float64 if torch.float64 in (student_logits.dtype, teacher_logits.dtype) else torch.float32
    student_log_probs = F.log_softmax(student_logits.to(dtype=dtype) / float(temperature), dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.detach().to(dtype=dtype) / float(temperature), dim=-1)
    contributions = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    unclipped = contributions.detach().sum(dim=-1)
    with torch.no_grad():
        student_entropy = -(student_log_probs.detach().exp() * student_log_probs.detach()).sum(dim=-1)
        teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1)
    if token_clip is None:
        clipped_fraction = torch.zeros_like(unclipped)
    else:
        clipped_fraction = (contributions.detach() > float(token_clip)).to(dtype=dtype).mean(dim=-1)
        contributions = contributions.clamp(max=float(token_clip))
    return {
        "response_forward_kl": contributions.sum(dim=-1),
        "response_forward_kl_unclipped": unclipped,
        "response_forward_kl_clipped_fraction": clipped_fraction.detach(),
        "response_student_entropy_nats": student_entropy,
        "response_teacher_entropy_nats": teacher_entropy,
    }


__all__ = ["forward_kl_token_values"]
