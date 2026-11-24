"""Shared Student-selected support for whole-sequence distribution losses."""
from __future__ import annotations

import torch

DEFAULT_KL_TOP_K = 512


def validate_kl_top_k(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("algorithm.kl_top_k must be a positive integer.")
    return value


def student_topk_logits(student: torch.Tensor, teacher: torch.Tensor | None = None,
                        *, top_k: int = DEFAULT_KL_TOP_K, return_indices: bool = False):
    """Select independently at every position, before any FP32 softmax.

    Selection is detached. Gather preserves the Student gradient and freezes
    the Teacher. Both distributions are subsequently normalized on these
    same token IDs. Small test vocabularies use min(k, vocabulary_size).
    """
    top_k = validate_kl_top_k(top_k)
    if student.ndim < 2 or not student.is_floating_point() or student.shape[-1] < 1:
        raise ValueError("Top-k KL requires floating logits with a nonempty vocabulary.")
    if teacher is not None and (teacher.shape != student.shape or teacher.device != student.device):
        raise ValueError("Student and Teacher logits must have equal shapes and devices.")
    if top_k >= student.shape[-1]:
        values = (student, None if teacher is None else teacher.detach())
        if return_indices:
            indices = torch.arange(student.shape[-1], device=student.device).expand(student.shape)
            return (*values, indices)
        return values
    indices = student.detach().topk(top_k, dim=-1, sorted=False).indices
    selected = student.gather(-1, indices)
    reference = None if teacher is None else teacher.detach().gather(-1, indices)
    return (selected, reference, indices) if return_indices else (selected, reference)


def sampled_log_prob(logits: torch.Tensor, response_ids: torch.Tensor, *, temperature=1.0) -> torch.Tensor:
    """True action probability for IS; independent of conditional KL support.

    A sampled action can lie outside Student top-k. Replacing its probability
    with a conditional probability would invalidate behavior-policy IS.
    This is a detached diagnostic, not a distribution-loss gradient.
    """
    with torch.no_grad():
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        values = logits.detach().to(dtype) / temperature
        sampled = values.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return sampled - values.logsumexp(-1)


def response_token_statistics(head, hidden, response_ids, required, *, kl_top_k=DEFAULT_KL_TOP_K):
    """One whole-response projection; action logprobs retain true normalization."""
    logits = head(hidden)
    result = {}
    topk_cfg = required.get("topk", {})
    want_topk = bool(topk_cfg.get("enabled", False))
    if required.get("response_logprobs", True) or want_topk:
        values = logits.float()
        normalization = values.logsumexp(-1)
        if required.get("response_logprobs", True):
            result["response_logprobs"] = values.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1) - normalization
        if want_topk:
            k = topk_cfg.get("top_k", 0)
            if type(k) is not int or k < 1 or k > logits.shape[-1]:
                raise ValueError("return_signals.topk.top_k must lie within the vocabulary size.")
            selected, indices = values.topk(k, dim=-1)
            result["topk_token_ids"] = indices.to(torch.int32)
            result["topk_logprobs"] = selected - normalization.unsqueeze(-1)
    if required.get("response_entropy", False):
        with torch.set_grad_enabled(torch.is_grad_enabled() and required.get("entropy_requires_grad", True)):
            selected, _ = student_topk_logits(logits, top_k=kl_top_k)
            logp = selected.float().log_softmax(-1)
            result["response_entropy"] = -(logp.exp() * logp).sum(-1)
    return result
