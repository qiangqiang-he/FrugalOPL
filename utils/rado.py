"""Student-top-k RADO targets and prompt-cluster direction estimates."""
from __future__ import annotations

import math
import torch
from utils.topk_kl import DEFAULT_KL_TOP_K, sampled_log_prob, student_topk_logits

RADO_KEYS = ("rado_kl_loss", "student_entropy_nats", "student_teacher_kl_pq",
             "rado_token_score", "student_sampled_logp", "teacher_sampled_logp")


def rado_token_values(student, teacher, response_ids, *, eta, temperature=1.,
                       top_k=DEFAULT_KL_TOP_K, **_):
    if not math.isfinite(eta) or temperature != 1.:
        raise ValueError("RADO requires a finite eta and temperature 1.")
    yp, yq = sampled_log_prob(student, response_ids), sampled_log_prob(teacher, response_ids)
    student, teacher, indices = student_topk_logits(student, teacher, top_k=top_k, return_indices=True)
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    lp = student.to(dtype).log_softmax(-1)
    with torch.no_grad():
        lq = teacher.to(dtype).log_softmax(-1)
        # No plausible-support mask: eta=0 is the conditional Student.
        lr = ((1 + eta) * lp.detach() - eta * lq).log_softmax(-1)
        r = lr.exp()
        p = lp.detach().exp()
        kl_pq = (p * (lp.detach() - lq)).sum(-1)
        # Reweight only the conditional support and preserve its total mass.
        # Actions outside the selected support have zero directional score;
        # their true action probabilities above still enter importance weights.
        membership = indices == response_ids.unsqueeze(-1)
        position = membership.long().argmax(-1, keepdim=True)
        conditional_log_ratio = (lp.detach() - lq).gather(-1, position).squeeze(-1)
        score = torch.where(membership.any(-1), conditional_log_ratio - kl_pq, 0.)
    return {"rado_kl_loss": (r * (lr - lp)).sum(-1),
            "student_entropy_nats": -(p * lp.detach()).sum(-1),
            "student_teacher_kl_pq": kl_pq, "rado_token_score": score,
            "student_sampled_logp": yp, "teacher_sampled_logp": yq}


def direction_estimate(scores, rewards, groups, log_weights, *, delta=.2,
                       ess_fraction=.5, confidence=.95, bootstrap_samples=4096,
                       seed=42, expected_group_size=8):
    """Unclipped trajectory IS, raw rewards, LOO, prompt-cluster bootstrap.

    A common positive exponential scale makes bootstrap sign/ESS stable.
    Reported G is restored to its original scale, not self-normalized IS.
    """
    s, r, g, lw = (torch.as_tensor(x, device='cpu').double().flatten()
                   for x in (scores, rewards, groups, log_weights))
    n = len(s)
    if n == 0 or any(len(x) != n for x in (r, g, lw)):
        raise ValueError("RADO score arrays must be nonempty and aligned.")
    if not all(torch.isfinite(x).all() for x in (s, r, g, lw)):
        raise ValueError("Nonfinite RADO score, reward or importance weight.")
    if not ((r == 0) | (r == 1)).all():
        raise ValueError("RADO rewards must be binary.")
    ids = torch.unique(g, sorted=True)
    baseline = torch.empty_like(r)
    for group in ids:
        mask = g == group
        k = int(mask.sum())
        if k != expected_group_size or k < 2:
            raise ValueError("RADO requires complete question groups with >=2 rollouts.")
        baseline[mask] = (r[mask].sum()-r[mask])/(k-1)
    scale = float(lw.max())
    w = (lw-scale).exp()
    ess = float(w.sum().square()/w.square().sum())
    # Reject numerically dangerous IS without silently clipping weights.
    safe_scale = abs(scale) < 600
    restore = math.exp(scale) if safe_scale else 1.
    x = w * (r-baseline) * s
    clusters = torch.stack([x[g == group].mean() for group in ids])
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(ids), (bootstrap_samples, len(ids)), generator=generator)
    sampled = clusters[draws].mean(-1)
    tail = (1-confidence)/2
    low, high = (float(v)*restore for v in torch.quantile(sampled, torch.tensor([tail, 1-tail], dtype=torch.float64)))
    estimate = float(x.mean())*restore
    direction = 1 if low > 0 else -1 if high < 0 else 0
    reason = 'reliable_direction'
    if not safe_scale or ess < ess_fraction*n:
        direction, reason = 0, 'low_ess' if safe_scale else 'unstable_importance_weights'
    elif direction == 0:
        reason = 'uncertain_direction'
    return {"eta": direction*delta, "direction": direction, "g_hat": estimate,
            "g_raw": float((w*r*s).mean())*restore, "g_ci_low": low, "g_ci_high": high,
            "g_bootstrap_se": float(sampled.std())*restore,
            "ess": ess, "ess_fraction": ess/n, "log_weight_max": float(lw.max()),
            "log_weight_min": float(lw.min()), "weight_mean": float(w.mean())*restore,
            "reward_is": float((w*r).mean())*restore,
            "reward_self_normalized": float((w*r).sum()/w.sum()),
            "correct_fraction": float(r.mean()), "score_mean": float(s.mean()),
            "prompt_count": len(ids), "rollout_count": n, "reason": reason}
