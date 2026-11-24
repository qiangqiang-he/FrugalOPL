"""Student-top-k detached TCDO targets; state is fixed for a trajectory."""
from __future__ import annotations

import torch
from utils.ratio_kl import ratio_kl_scores
from utils.topk_kl import DEFAULT_KL_TOP_K, student_topk_logits

CORRECT, INCORRECT, TRUNCATED = 0, 1, 2
TCDO_KEYS = ("tcdo_kl_loss", "student_entropy_nats", "target_entropy_nats", "active_teacher_forward_kl")


def tcdo_token_values(student, teacher, *, state, temperature=1.0, plausible_fraction=0.01,
                      top_k=DEFAULT_KL_TOP_K):
    if state == INCORRECT:
        values = ratio_kl_scores(student, teacher, temperature=temperature,
                                 plausible_fraction=plausible_fraction, top_k=top_k)
        return {"tcdo_kl_loss": values["ratio_kl_loss"], "student_entropy_nats": values["student_entropy_nats"],
                "target_entropy_nats": values["target_entropy_nats"], "active_teacher_forward_kl": values["teacher_forward_kl"]}
    student, teacher = student_topk_logits(student, teacher, top_k=top_k)
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    logp = (student.to(dtype) / temperature).log_softmax(-1)
    with torch.no_grad():
        entropy = -(logp.exp() * logp).sum(-1)
    if state == CORRECT:
        return {"tcdo_kl_loss": logp.sum(-1) * 0, "student_entropy_nats": entropy,
                "target_entropy_nats": entropy}
    if state != TRUNCATED:
        raise ValueError("Unknown TCDO state")
    with torch.no_grad():
        logq = (teacher.detach().to(dtype) / temperature).log_softmax(-1)
        q = logq.exp()
    loss = (q * (logq - logp)).sum(-1)
    return {"tcdo_kl_loss": loss, "student_entropy_nats": entropy,
            "target_entropy_nats": -(q * logq).sum(-1), "active_teacher_forward_kl": loss.detach()}
