"""Self-distillation onto the old top-p head within Student top-k."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from utils.topk_kl import DEFAULT_KL_TOP_K, student_topk_logits


def validate_tail_sup_parameters(top_p: float, temperature: float) -> tuple[float, float]:
    for name, value in (("top_p", top_p), ("temperature", temperature)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Tail suppression {name} must be a finite number.")
        if not math.isfinite(value) or value <= 0 or (name == "top_p" and value >= 1):
            raise ValueError("Tail suppression requires 0 < top_p < 1 and temperature > 0.")
    return float(top_p), float(temperature)


def tail_suppression_token_values(
    student_logits: torch.Tensor,
    old_logits: torch.Tensor,
    *,
    top_p: float = 0.99,
    temperature: float = 1.0,
    top_k: int = DEFAULT_KL_TOP_K,
) -> dict[str, torch.Tensor]:
    """Return exact KL(q_TS || p_student) and detached position diagnostics.

    Include the token crossing the cumulative threshold. A position is
    inactive only when its old maximum probability is strictly above top_p.
    Both softmaxes normalize over Student top-k; there is no clipping
    or T-squared multiplier. The target, head and validity mask are detached.
    The runtime checkpoints one whole-response head/loss call.
    """

    top_p, temperature = validate_tail_sup_parameters(top_p, temperature)
    if not isinstance(student_logits, torch.Tensor) or not isinstance(old_logits, torch.Tensor):
        raise TypeError("Tail suppression logits must be tensors.")
    if student_logits.ndim < 2 or student_logits.shape != old_logits.shape or student_logits.size(-1) < 1:
        raise ValueError("Tail suppression requires equal logits shapes and a non-empty vocabulary.")
    if not student_logits.is_floating_point() or not old_logits.is_floating_point():
        raise TypeError("Tail suppression logits must be floating point.")
    if student_logits.device != old_logits.device:
        raise ValueError("Tail suppression logits must be on the same device.")
    student_logits, old_logits = student_topk_logits(student_logits, old_logits, top_k=top_k)
    dtype = torch.float64 if torch.float64 in (student_logits.dtype, old_logits.dtype) else torch.float32
    with torch.no_grad():
        old_logp = F.log_softmax(old_logits.detach().to(dtype=dtype) / temperature, dim=-1)
        probabilities, indices = old_logp.exp().sort(dim=-1, descending=True, stable=True)
        cumulative = probabilities.cumsum(-1)
        previous_mass = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), dim=-1)
        head = previous_mass < top_p
        head_mass = (probabilities * head).sum(-1)
        target = probabilities * head / head_mass.unsqueeze(-1)
        # logq outside the head is unused; avoid log(0) and 0 * -inf.
        logq = torch.where(head, old_logp.gather(-1, indices) - head_mass.log().unsqueeze(-1), 0.0)
        active = probabilities[..., 0] <= top_p

    logp = F.log_softmax(student_logits.to(dtype=dtype) / temperature, dim=-1)
    sorted_logp = logp.gather(-1, indices)
    terms = torch.where(head, target * (logq - sorted_logp), 0.0)
    kl = torch.where(active, terms.sum(-1), 0.0)
    with torch.no_grad():
        current = logp.detach().exp()
        tail_mass = (current.gather(-1, indices) * ~head).sum(-1)
        entropy = -(current * logp.detach()).sum(-1)
    return {
        "response_tail_sup_kl": kl,
        "response_tail_sup_active": active,
        "response_tail_sup_head_mass": head_mass.detach(),
        "response_tail_sup_head_size": head.sum(-1).detach(),
        "response_tail_sup_old_tail_mass": (1.0 - head_mass).clamp_min(0).detach(),
        "response_tail_sup_tail_mass": tail_mass,
        "response_tail_sup_entropy": entropy,
        "response_tail_sup_max_probability": probabilities[..., 0].detach(),
    }


__all__ = ["tail_suppression_token_values", "validate_tail_sup_parameters"]
