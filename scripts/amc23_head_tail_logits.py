"""Condition the first few new tokens of each paired continuation."""

import json
from functools import partial

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def token_masks(logits: torch.Tensor, tail_threshold: float | None = None):
    sorted_logits, token_ids = logits.sort(descending=True, stable=True)
    cumulative = sorted_logits.softmax(dim=0, dtype=torch.float64).cumsum(dim=0)
    # The first token that takes cumulative mass strictly above 0.9 is included.
    previous_mass = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    head = previous_mass <= 0.9
    tail = ~head
    if tail_threshold is not None:
        tail = tail & (previous_mass <= tail_threshold)
    return head.scatter(0, token_ids, head), tail.scatter(0, token_ids, tail)


class HeadTailLogitsProcessor(AdapterLogitsProcessor):
    """Share one full-vocabulary distribution between the two branches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = {}

    @classmethod
    def validate_params(cls, params):
        arguments = params.extra_args or {}
        if arguments and (arguments.get("branch") not in {"head", "tail"} or "pair" not in arguments):
            raise ValueError("Expected a pair ID and a head or tail branch")

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        if not params.extra_args:
            return None
        return partial(self.condition, params.extra_args)

    def condition(self, arguments, output_ids, logits):
        step = len(output_ids) + 1
        if step > arguments.get("actions", 1):
            return logits
        pair, branch = arguments["pair"], arguments["branch"]
        tail_threshold = arguments.get("tail_threshold")
        if step == 1:
            previous = self.pending.pop(pair, None)
            if previous is None:
                original = logits.clone()
                head, tail = token_masks(original, tail_threshold)
                self.pending[pair] = (original, head, tail, branch)
            else:
                original, head, tail, first_branch = previous
                if first_branch == branch:
                    raise RuntimeError(f"Duplicate branch {branch} for prefix {pair}")
        else:
            original = logits
            head, tail = token_masks(original, tail_threshold)
        keep = head if branch == "head" else tail
        if branch == "tail" and tail_threshold is not None and not bool(keep.any()):
            # End the engine request and surface the empty action set to the
            # caller. The synthetic stop token is never a rollout action.
            with open(arguments["empty_tail_log"], "a", encoding="utf-8") as log:
                log.write(json.dumps({"pair": pair, "step": step}) + "\n")
            stopped = torch.full_like(original, -torch.inf)
            stopped[arguments["stop_token"]] = 0
            return stopped
        return original.masked_fill(~keep, -torch.inf)
