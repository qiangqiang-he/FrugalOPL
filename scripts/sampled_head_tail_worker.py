"""vLLM 0.19.1 worker instrumentation; no changes to sampling or model weights."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from utils.sampled_head_tail import METRICS, sampled_partition_metrics


class SampledHeadTailWorker:
    def install_sampled_head_tail(self, vocab_size: int, scratch_dir: str):
        runner = self.model_runner
        if runner.use_async_scheduling or runner.num_spec_tokens:
            raise ValueError("Statistics require sync scheduling and no speculative decoding")
        if hasattr(runner, "_sampled_head_tail_state"):
            raise RuntimeError("Instrumentation already installed")
        state = {"active": False, "pending": [], "metadata": [], "chunks": [], "rows": 0,
                 "vocab": vocab_size, "scratch": scratch_dir, "last_progress": 0.}
        runner._sampled_head_tail_state = state
        original = runner._sample

        def instrumented(logits, spec_decode_metadata):
            if not state["active"]:
                return original(logits, spec_decode_metadata)
            if spec_decode_metadata is not None:
                raise RuntimeError("Speculative decoding would invalidate position accounting")
            num = runner.input_batch.num_reqs
            keep = np.flatnonzero(~runner.discard_request_mask.np[:num].astype(bool))
            # FP32 Sampler mutates its inputs in place; preserve before sampling.
            full_logits = logits[:num, :vocab_size].float().clone()
            metadata = []
            for i in keep:
                req = runner.requests[runner.input_batch.req_ids[i]]
                params = req.sampling_params
                if (params.temperature != 1. or params.top_p != 1. or params.top_k != -1
                    or params.repetition_penalty != 1. or params.presence_penalty != 0.
                    or params.frequency_penalty != 0. or params.min_tokens != 0
                    or params.bad_words or params.allowed_token_ids or params.logit_bias):
                    raise ValueError("Only unmodified full-vocabulary T=1 sampling is supported")
                metadata.append((params.extra_args["head_tail_row"], len(req.output_token_ids) + 1))
            output = original(logits, spec_decode_metadata)
            if len(keep):
                idx = torch.as_tensor(keep, device=full_logits.device)
                ids = output.sampled_token_ids[idx, 0]
                values = sampled_partition_metrics(full_logits[idx], ids)
                rows = torch.cat((torch.as_tensor(metadata, device=ids.device, dtype=torch.float32),
                                  ids[:, None].float(), values), 1)
                state["pending"].append(rows)
                state["rows"] += len(keep)
                if sum(len(x) for x in state["pending"]) >= 4096:
                    self.flush_sampled_head_tail()
            now = time.monotonic()
            if now - state["last_progress"] > 30:
                state["last_progress"] = now
                print(f"[HeadTail GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] sampled_positions_in_batch={state['rows']}", flush=True)
            return output

        runner._sample = instrumented
        return {"vocab_size": vocab_size, "metrics": list(METRICS), "pid": os.getpid()}

    def flush_sampled_head_tail(self):
        state = self.model_runner._sampled_head_tail_state
        if state["pending"]:
            state["chunks"].append(torch.cat(state["pending"]).cpu().numpy())
            state["pending"].clear()

    def begin_sampled_head_tail(self, batch_id: int):
        state = self.model_runner._sampled_head_tail_state
        if state["active"] or state["chunks"] or state["pending"]:
            raise RuntimeError("Previous statistics batch was not drained")
        state.update(active=True, rows=0, batch=batch_id)
        return True

    def end_sampled_head_tail(self):
        state = self.model_runner._sampled_head_tail_state
        self.flush_sampled_head_tail()
        if not state["rows"]:
            raise RuntimeError("No generated positions observed")
        path = Path(state["scratch"]) / f"batch-{state['batch']:09d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.concatenate(state["chunks"])
        np.save(path, array, allow_pickle=False)
        state["chunks"].clear()
        state["active"] = False
        return {"path": str(path), "rows": len(array)}
