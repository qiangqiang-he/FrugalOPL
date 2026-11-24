"""Native model loading and the output-head transform used by token losses."""
from __future__ import annotations

from collections.abc import Mapping
import torch


def policy_trust_remote_code(policy: Mapping[str, object]) -> bool:
    value = policy.get("trust_remote_code", True)
    if type(value) is not bool:
        raise TypeError("run.policy.trust_remote_code must be a boolean.")
    return value


def policy_model_factory(policy: Mapping[str, object]):
    """Keep causal-LM loading unchanged unless a multimodal profile opts in."""
    loader = policy.get("model_loader", "causal_lm")
    if loader == "causal_lm":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    if loader == "multimodal_lm":
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    raise ValueError("run.policy.model_loader must be causal_lm or multimodal_lm.")


def expanded_gqa_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, is_causal=None, **kwargs):
    """Equivalent GQA using equal head counts, including 512-wide Gemma heads.

    Torch's efficient CUDA kernel accepts the expanded heads but not the
    implicit GQA dispatch. Expansion is linear in sequence length and its
    backward sums the replicated K/V gradients; no score matrix is built here.
    The Learner's existing context still forbids the math SDPA backend.
    """
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1]:
            raise ValueError("Query heads must be divisible by key/value heads.")
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=causal, enable_gqa=False,
    )
    return output.transpose(1, 2).contiguous(), None


def policy_sdpa_implementation(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", config)
    if getattr(text, "model_type", None) != "gemma4_text":
        return "sdpa"
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
    name = "frugalopl_expanded_gqa_sdpa"
    AttentionInterface.register(name, expanded_gqa_sdpa)
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return name


class SoftcappedOutputHead(torch.nn.Module):
    """Apply the native Gemma output transform before any probability/loss."""

    def __init__(self, head: torch.nn.Module, cap: float):
        super().__init__()
        self.head = head
        self.cap = cap
        self.training = head.training

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return (self.head(hidden) / self.cap).tanh() * self.cap


def model_output_head(module: torch.nn.Module, *, worker: str):
    getter = getattr(module, "get_base_model", None)
    base = getter() if callable(getter) else None
    for candidate in (base, module):
        if candidate is None:
            continue
        embeddings = getattr(candidate, "get_output_embeddings", None)
        head = embeddings() if callable(embeddings) else None
        if not callable(head):
            head = getattr(candidate, "lm_head", None)
        if callable(head):
            config = getattr(candidate, "config", getattr(module, "config", None))
            config = getattr(config, "text_config", config)
            cap = getattr(config, "final_logit_softcapping", None)
            if cap is not None:
                cap = float(cap)
                if not cap > 0:
                    raise ValueError("final_logit_softcapping must be positive when configured.")
                return SoftcappedOutputHead(head, cap)
            return head
    raise RuntimeError(f"{worker} model must expose an lm_head.")
