"""Detached p * (p/q)^n targets on Student top-k with plausible filtering."""

from __future__ import annotations

import math

import torch
from utils.topk_kl import DEFAULT_KL_TOP_K, sampled_log_prob, student_topk_logits


RATIO_KEYS = ("ratio_kl_loss", "teacher_forward_kl", "student_entropy_nats", "teacher_entropy_nats",
              "target_entropy_nats", "target_max_probability", "target_mass_error", "plausible_vocabulary_fraction")


def ratio_kl_scores(student_logits, teacher_logits, response_ids=None, *, temperature=1.0,
                    plausible_fraction=0.1, top_k=DEFAULT_KL_TOP_K, ratio_power=1.0):
    """KL(stopgrad(normalize(p * (p/q)^ratio_power * support)) || p).

    The support is p_i >= plausible_fraction * max(p). No CLIP, reward,
    sampled-token approximation, or additional prefix weight is applied.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("ratio_temperature must be positive and finite.")
    if not math.isfinite(plausible_fraction) or not 0 < plausible_fraction <= 1:
        raise ValueError("ratio_plausible_fraction must lie in (0, 1].")
    if isinstance(ratio_power, bool) or not isinstance(ratio_power, (int, float)) or not math.isfinite(ratio_power):
        raise ValueError('ratio_power must be a finite real number.')
    sampled = None if response_ids is None else sampled_log_prob(student_logits, response_ids, temperature=temperature)
    student_logits, teacher_logits = student_topk_logits(student_logits, teacher_logits, top_k=top_k)
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_p = (student_logits.to(dtype) / temperature).log_softmax(-1)
    with torch.no_grad():
        log_q = (teacher_logits.to(dtype) / temperature).log_softmax(-1)
        support = log_p >= log_p.amax(-1, keepdim=True) + math.log(plausible_fraction)
        target = ((1 + ratio_power) * log_p - ratio_power * log_q).masked_fill(~support, -torch.inf).softmax(-1)
        target_xlogx = torch.special.xlogy(target, target).sum(-1)
    loss = target_xlogx - (target * log_p).sum(-1)
    with torch.no_grad():
        p, q = log_p.exp(), log_q.exp()
        metrics = {
            "teacher_forward_kl": (q * (log_q - log_p)).sum(-1),
            "student_entropy_nats": -(p * log_p).sum(-1),
            "teacher_entropy_nats": -(q * log_q).sum(-1),
            "target_entropy_nats": -target_xlogx,
            "target_max_probability": target.amax(-1),
            "target_mass_error": (target.sum(-1) - 1).abs(),
            "plausible_vocabulary_fraction": support.to(dtype).mean(-1),
        }
        if response_ids is not None:
            metrics["student_sampled_logp_nats"] = sampled
    return {"ratio_kl_loss": loss, **metrics}
