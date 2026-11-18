"""Full-vocabulary OCSD loss and probability transfer at temperature one."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_ocsd_parameters(gamma: object, tau: object) -> tuple[float, float]:
    for name, value in (("ocsd_gamma", gamma), ("ocsd_tau", tau)):
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f"algorithm.{name} must be a number in [0, 1].")
    return float(gamma), float(tau)


def ocsd_token_values(
    logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    correct: torch.Tensor,
    *,
    gamma: float,
    tau: float,
) -> dict[str, torch.Tensor]:
    """Return token KL and detached diagnostics for a chunk of response positions.

    Logits are [batch, tokens, vocab], sampled_ids are [batch, tokens], and
    correct is one boolean per rollout. The two KL sides share one forward.
    """
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    logp = F.log_softmax(logits, dim=-1, dtype=dtype)
    with torch.no_grad():
        detached_logits = logits.detach()
        sampled_logits = detached_logits.gather(-1, sampled_ids.unsqueeze(-1))
        # Softmax preserves order. Using logits preserves that order even when
        # very small probabilities underflow to zero.
        head = detached_logits >= sampled_logits
        probabilities = logp.detach().exp()
        positive = correct[:, None, None]
        eligible_head = head & ~(positive & (probabilities > tau))
        head_mass = (probabilities * eligible_head).sum(-1, keepdim=True)
        tail_mass = (probabilities * ~head).sum(-1, keepdim=True)
        transfer = gamma * torch.minimum(head_mass, tail_mass)

        direction = torch.where(positive, 1.0, -1.0)
        head_change = direction * transfer / torch.where(head_mass > 0, head_mass, 1.0)
        tail_change = -direction * transfer / torch.where(tail_mass > 0, tail_mass, 1.0)
        relative_change = torch.where(eligible_head, head_change, torch.where(head, 0.0, tail_change))
        target = probabilities * (1.0 + relative_change)
        has_mass = target > 0
        # q follows q = p * (1 + relative_change). log1p preserves very small
        # transfers, and a zero-mass term contributes zero even when gamma=1.
        log_ratio = torch.where(has_mass, relative_change, 0.0).log1p()
        total_head_mass = (probabilities * head).sum(-1)

    safe_logp = logp.masked_fill(~has_mass, 0.0)
    # log(q) - logp = log(q / p.detach()) + logp.detach() - logp.
    kl = (target * (log_ratio + (safe_logp.detach() - safe_logp))).sum(-1)
    transfer = transfer.squeeze(-1)
    # q == p here, so both the KL and its derivative are exactly zero.
    kl = torch.where(transfer > 0, kl, 0.0)
    return {
        "response_ocsd_kl": kl,
        "response_ocsd_head_mass": total_head_mass,
        "response_ocsd_transfer": transfer,
    }
