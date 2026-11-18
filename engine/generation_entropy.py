"""Collect full-vocabulary rollout entropy inside the vLLM worker."""

from __future__ import annotations

import torch


ENTROPY_GROUP = "frugalopl_entropy_group"


class GenerationEntropyWorker:
    def install_generation_entropy(self) -> None:
        runner = self.model_runner
        if runner.use_async_scheduling or runner.num_spec_tokens:
            raise RuntimeError("Generation entropy requires synchronous, non-speculative sampling.")
        self._entropy_sums = None
        self._entropy_counts = None
        sample = runner._sample
        vocab_size = runner.model_config.get_vocab_size()

        def sample_with_entropy(logits, spec_decode_metadata):
            if self._entropy_sums is not None:
                batch = runner.input_batch
                rows, groups = [], []
                discarded = runner.discard_request_mask.np
                for row in range(batch.num_reqs):
                    if discarded[row]:
                        continue
                    params = runner.requests[batch.req_ids[row]].sampling_params
                    group = (params.extra_args or {}).get(ENTROPY_GROUP)
                    if group is not None:
                        rows.append(row)
                        groups.append(group)
                        self._entropy_counts[group] += 1
                if rows:
                    indices = torch.tensor(rows, device=logits.device)
                    probabilities = logits[:, :vocab_size].index_select(0, indices).softmax(-1, dtype=torch.float32)
                    entropy = torch.special.entr(probabilities).sum(-1)
                    # The GPU keeps scalar sums only. Sample uses the original logits.
                    group_ids = torch.tensor(groups, device=logits.device)
                    self._entropy_sums.index_add_(0, group_ids, entropy.double())
            return sample(logits, spec_decode_metadata)

        runner._sample = sample_with_entropy

    def begin_generation_entropy(self, group_count: int) -> None:
        if self._entropy_sums is not None:
            raise RuntimeError("Generation entropy for the previous call was not consumed.")
        self._entropy_sums = torch.zeros(group_count, device=self.model_runner.device, dtype=torch.float64)
        self._entropy_counts = [0] * group_count

    def finish_generation_entropy(self) -> list[tuple[float, int]]:
        sums, counts = self._entropy_sums, self._entropy_counts
        self._entropy_sums = None
        self._entropy_counts = None
        return list(zip(sums.tolist(), counts, strict=True))
