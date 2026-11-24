"""Per-card Actor/Learner handoff with a shared, one-rollout work queue.

Sampling and local backwards overlap; parameters change only once, after the
complete policy-versioned global batch has been consumed by all Learners.
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from algorithm_plugins.algorithm_base import ObjectiveResult


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def sum_gradient_buffers(gradients, *, bucket_bytes=32 * 1024**2):
    """SUM optimizer-consumed buffers once, regardless of per-rank work count."""
    if not _distributed():
        return
    buckets, current, size, kind = [], [], 0, None
    for gradient in gradients:
        next_kind = (gradient.dtype, gradient.device)
        if current and (kind != next_kind or size + gradient.numel() * gradient.element_size() > bucket_bytes):
            buckets.append(current)
            current, size = [], 0
        kind = next_kind
        current.append(gradient)
        size += gradient.numel() * gradient.element_size()
    if current:
        buckets.append(current)
    with torch.no_grad():
        for bucket in buckets:
            packed = torch.cat([gradient.reshape(-1) for gradient in bucket])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            offset = 0
            for gradient in bucket:
                gradient.copy_(packed[offset:offset + gradient.numel()].reshape_as(gradient))
                offset += gradient.numel()


class StreamingAccumulator:
    """Local objective accumulation; contains no collectives until finish()."""
    def __init__(self, learner, expected_rollouts, token_bound):
        self.learner = learner
        self.expected = expected_rollouts
        self.token_bound = token_bound
        if expected_rollouts < 1 or token_bound < 1:
            raise ValueError("Streaming accumulation requires positive global denominators.")
        self.token_mode = learner.loss_aggregation == "token"
        self.stats = dict(objective_sum=0., objective_weight=0., valid_tokens=0,
                          update_tokens=0, counted_rollouts=0, metric_sums={}, metric_weights={},
                          advantage_count=0, advantage_sum=0., advantage_square_sum=0.)
        self.first_backward_at = 0.
        self.last_backward_at = 0.
        self.compute_intervals = []
        learner._zero_grad()

    def consume(self, micro):
        started = time.monotonic()
        with self.learner._bounded_sdpa_context():
            self._consume(micro)
        self.compute_intervals.append((started, time.monotonic()))

    def _consume(self, micro):
        from engine.learner import _learner_plugin_scope, _metric_number, _move_tensors_to_device
        learner = self.learner
        batch = _move_tensors_to_device(micro, learner.device)
        if batch["response_ids"].ndim != 2 or batch["response_ids"].size(0) != 1:
            raise ValueError("Streaming Learners consume exactly one rollout at a time.")
        with learner._bounded_sdpa_context(), _learner_plugin_scope(learner, phase="objective"):
            computed = getattr(learner.plugin, learner.objective_fn)(batch)
        if not isinstance(computed, ObjectiveResult):
            raise TypeError("Streaming Learners require ObjectiveResult with exact token counts.")
        objective, metrics = computed
        if not isinstance(objective, torch.Tensor) or objective.numel() != 1 or not objective.requires_grad:
            raise ValueError("Streaming objectives must be differentiable scalar tensors.")
        if not isinstance(computed.requires_optimizer_step, bool) or not isinstance(metrics, Mapping):
            raise TypeError("Invalid streaming objective update flag or metrics.")
        tokens = learner._objective_valid_token_count(computed, batch)
        weight = float(tokens) if self.token_mode else 1.
        denominator = self.token_bound if self.token_mode else self.expected
        if computed.requires_optimizer_step:
            learner._backward(-objective * (weight / denominator))
        elif float(objective.detach()) != 0.:
            raise ValueError("An objective without an optimizer update must be zero.")
        now = time.monotonic()
        if not self.first_backward_at:
            self.first_backward_at = now
        self.last_backward_at = now
        stats = self.stats
        stats["valid_tokens"] += tokens
        stats["update_tokens"] += tokens if computed.requires_optimizer_step else 0
        stats["counted_rollouts"] += 1
        stats["objective_sum"] += float(objective.detach().float()) * weight
        stats["objective_weight"] += weight
        if "advantages" in batch:
            advantage = torch.as_tensor(batch["advantages"]).detach().double()
            stats["advantage_count"] += advantage.numel()
            stats["advantage_sum"] += float(advantage.sum())
            stats["advantage_square_sum"] += float(advantage.square().sum())
        for name, value in metrics.items():
            if name == "objective":
                continue
            numeric = _metric_number(value)
            if numeric is None:
                continue
            metric_weight = weight if self.token_mode and (name == "loss" or name.endswith("_loss")) else 1.
            if name in computed.token_weighted_metrics or name.endswith("_token_fraction"):
                metric_weight = float(tokens)
            stats["metric_sums"][name] = stats["metric_sums"].get(name, 0.) + numeric * metric_weight
            stats["metric_weights"][name] = stats["metric_weights"].get(name, 0.) + metric_weight

    def finish(self):
        """Called only after every rank owns its GPU and the queue has drained."""
        learner = self.learner
        stats_by_rank = [self.stats]
        if _distributed():
            stats_by_rank = [None] * learner.world_size
            dist.all_gather_object(stats_by_rank, self.stats, group=learner._control_group)
        total = sum(item["counted_rollouts"] for item in stats_by_rank)
        if total != self.expected:
            raise RuntimeError(f"Streaming work count mismatch: {total} != {self.expected}.")
        valid_tokens = sum(item["valid_tokens"] for item in stats_by_rank)
        update_tokens = sum(item["update_tokens"] for item in stats_by_rank)
        # Touch all trainable parameters once, even on a rank assigned zero
        # samples, so every optimizer exposes the same ordered gradient set.
        with learner.model_engine.no_sync() if learner.model_engine is not None else nullcontext():
            learner._backward(learner._dummy_loss())
        gradients, _ = learner._optimizer_gradient_state(strict=True)
        sum_gradient_buffers(gradients)
        if self.token_mode and valid_tokens:
            with torch.no_grad():
                for gradient in gradients:
                    gradient.mul_(self.token_bound / valid_tokens)
        advance = bool(valid_tokens) if self.token_mode else True
        update = advance and (bool(update_tokens) if self.token_mode else True)
        raw = learner._gradient_norm() if update else 0.
        if update:
            learner._clip_gradients()
        clipped = learner._gradient_norm() if update else 0.
        if update:
            learner._step()
        else:
            learner._zero_grad()
        if advance:
            learner.global_step += 1
        sums, weights = {}, {}
        for item in stats_by_rank:
            for key, value in item["metric_sums"].items():
                sums[key] = sums.get(key, 0.) + value
                weights[key] = weights.get(key, 0.) + item["metric_weights"][key]
        metrics = {key: sums[key] / weights[key] for key in sorted(sums) if weights[key] > 0}
        advantage_count = sum(item["advantage_count"] for item in stats_by_rank)
        if advantage_count:
            mean = sum(item["advantage_sum"] for item in stats_by_rank) / advantage_count
            variance = sum(item["advantage_square_sum"] for item in stats_by_rank) / advantage_count - mean**2
            if "grpo_advantage_mean" in metrics:
                metrics["grpo_advantage_mean"] = mean
            if "grpo_advantage_std" in metrics:
                metrics["grpo_advantage_std"] = math.sqrt(max(0., variance))
        objective_weight = sum(item["objective_weight"] for item in stats_by_rank)
        metrics["loss"] = -sum(item["objective_sum"] for item in stats_by_rank) / objective_weight if objective_weight else 0.
        if advance:
            learner._complete_algorithm_training_step(metrics)
        adapter = learner._maybe_publish_adapter() if advance else None
        if (update and learner.teacher_sync_interval > 0
                and learner.global_step % learner.teacher_sync_interval == 0 and _distributed()):
            dist.barrier()
        framework = dict(train_rollouts=float(total), micro_batches=float(total), kl_top_k=float(learner.kl_top_k),
                         optimizer_steps=float(update), logical_steps=float(advance), lr=learner._current_lr(),
                         raw_gradient=raw, clip_gradient=clipped, grad_norm=raw,
                         loss_aggregation=learner.loss_aggregation, objective_valid_tokens=float(valid_tokens))
        if adapter:
            framework["lora_path"] = adapter
        return metrics, framework


def prepare_group(learner, shard):
    """Compute group-dependent signals once, before individual samples queue."""
    from engine.learner import _learner_plugin_scope
    prepared = learner.plugin.prepare_learner_batch(dict(shard))
    prepared.pop("actor_shard", None)
    prepared.pop("question_accuracies", None)
    logical = learner._normalize_logical_signals(prepared)
    previous = getattr(learner, "_local_signal_collection", False)
    learner._local_signal_collection = True
    try:
        with learner._bounded_sdpa_context(), _learner_plugin_scope(learner, phase="collect"):
            additions = getattr(learner.plugin, learner.plugin_hook)(copy.deepcopy(logical))
    finally:
        learner._local_signal_collection = previous
    if not isinstance(additions, Mapping) or set(additions) & set(logical):
        raise ValueError("Streaming signal hooks must return new fields only.")
    return learner.build_micro_batches({**logical, **additions})


def run_overlapped_learner(learner, stop_event):
    from engine.learner import _deep_cpu_detach, _run_name_and_redis
    from utils.algorithm_runtime import resolve_train_candidate_count
    client = learner._redis_client()
    name, prefix = _run_name_and_redis(learner.config)
    root = f"{prefix}:{name}"
    algorithm = learner.config["algorithm"]
    questions = resolve_train_candidate_count(learner.config["controller"]["train_prompt_batch_size"], algorithm)
    expected = questions * algorithm["training_sampling"]["n"]
    shard_count = min(questions, len(learner.scheduling_plan.actor_gpu_ids))
    token_bound = expected * int(algorithm["training_sampling"]["max_tokens"])
    ds_cfg = learner.deepspeed_config()
    if ds_cfg.get("zero_optimization", {}).get("stage", 0) != 0 or not ds_cfg.get("managed_gradient_accumulation", True):
        raise ValueError("Overlapped Learners require ZeRO stage 0 and managed gradient accumulation.")
    if not callable(getattr(learner.model_engine, "no_sync", None)):
        raise RuntimeError("Overlapped Learners require DeepSpeed no_sync().")
    last_round = 0
    try:
        while not stop_event.is_set():
            manifest = learner.round_gate.wait_local_learner(learner.gpu_id, last_round, stop_event, timeout=1.)
            if manifest is None:
                continue
            if len(manifest.batch_ids) != 1 or manifest.policy_version != learner.global_step:
                raise RuntimeError("Streaming Learner received a stale policy or multi-batch round.")
            if not learner.gpu_gate.acquire(stop_event):
                return
            started = time.monotonic()
            pool = f"{root}:overlap:{manifest.round_id}"
            success = False
            try:
                model_load_seconds = learner._activate_model()
                accumulation = StreamingAccumulator(learner, expected, token_bound)
                with learner.model_engine.no_sync():
                    while not stop_event.is_set():
                        raw = client.rpop(f"{pool}:ready")
                        if raw is not None:
                            task_id, micro = pickle.loads(raw)
                            if not client.hsetnx(f"{pool}:claimed", task_id, learner.rank):
                                raise RuntimeError("Duplicate streaming sample claim.")
                            accumulation.consume(micro)
                            client.incr(f"{pool}:done")
                            continue
                        raw_group = client.rpop(f"{pool}:groups")
                        if raw_group is not None:
                            shard = pickle.loads(raw_group)
                            if (shard.get("round_id") != manifest.round_id
                                    or shard.get("behavior_version") != manifest.policy_version
                                    or shard.get("batch_id") != manifest.batch_ids[0]):
                                raise RuntimeError("Streaming group does not match its policy-versioned round.")
                            index = shard["actor_shard"]["index"]
                            if not client.hsetnx(f"{pool}:prepared_ids", index, learner.rank):
                                raise RuntimeError("Duplicate streaming question group.")
                            micro_batches = prepare_group(learner, shard)
                            if len(micro_batches) != len(shard["questions"]) * algorithm["training_sampling"]["n"]:
                                raise RuntimeError("Streaming plugins must preserve complete question groups.")
                            transaction = client.pipeline()
                            for index_in_shard, micro in enumerate(micro_batches):
                                transaction.lpush(f"{pool}:ready", pickle.dumps((f"{index}:{index_in_shard}",
                                                  _deep_cpu_detach(micro)), protocol=pickle.HIGHEST_PROTOCOL))
                            transaction.incr(f"{pool}:prepared")
                            transaction.execute()
                            continue
                        prepared, done = [int(value or 0) for value in client.mget(f"{pool}:prepared", f"{pool}:done")]
                        if prepared > shard_count or done > expected:
                            raise RuntimeError("Streaming pool exceeded its exact batch manifest.")
                        if prepared == shard_count and done == expected:
                            break
                        stop_event.wait(.01)
                if stop_event.is_set():
                    return
                if not learner.round_gate.learner_ready(learner.rank, manifest.round_id, stop_event, wait_for_actors=True):
                    return
                metrics, framework = accumulation.finish()
                report = dict(rank=learner.rank, gpu_id=learner.gpu_id,
                              rollouts=accumulation.stats["counted_rollouts"],
                              first_backward_at=accumulation.first_backward_at,
                              last_backward_at=accumulation.last_backward_at,
                              compute_intervals=accumulation.compute_intervals,
                              actor_finished_at=learner.round_gate.actor_finish_times(manifest.round_id))
                reports = [report]
                if _distributed():
                    reports = [None] * learner.world_size
                    dist.all_gather_object(reports, report, group=learner._control_group)
                last_actor = max(report["actor_finished_at"].values())
                overlap_seconds = sum(max(0., min(end, last_actor) - start)
                                      for row in reports for start, end in row["compute_intervals"])
                framework.update(actor_learner_overlap_seconds=overlap_seconds,
                                 per_rank_rollout_counts={str(row["rank"]): row["rollouts"] for row in reports})
                learner._synchronize_cuda()
                model_offload_seconds = learner._offload_model()
                if learner.rank == 0:
                    batches = learner._collect_round_batches(client, manifest, stop_event)
                    if not batches:
                        raise RuntimeError("Missing final global Actor batch for streaming telemetry.")
                    full = batches[0]
                    accuracies = full.get("question_accuracies") if learner.save_question_accuracy else None
                    result = learner._training_result(full, metrics, framework, rollout_count=expected,
                                                      question_accuracies=accuracies)
                    result = learner._with_worker_timing(result, worker=learner.node_id,
                        processing_seconds=time.monotonic() - started, gate_wait_seconds=0.,
                        model_load_seconds=model_load_seconds, model_offload_seconds=model_offload_seconds)
                    output = Path(learner.run_cfg.get("storage", {}).get("output_root", "outputs")) / name
                    audit = {"round_id": manifest.round_id, "policy_version": manifest.policy_version,
                             "global_step": learner.global_step, "expected_rollouts": expected,
                             "claimed_samples": {key.decode(): int(value) for key, value in client.hgetall(f"{pool}:claimed").items()},
                             "ranks": reports, "overlap_seconds": overlap_seconds}
                    (output / "overlap").mkdir(parents=True, exist_ok=True)
                    (output / "overlap" / f"step_{learner.global_step:06d}.json").write_text(json.dumps(audit), encoding="utf-8")
                    transaction = client.pipeline()
                    transaction.lpush(learner.queue_keys()["controller_train_results"],
                                      pickle.dumps(_deep_cpu_detach(result), protocol=pickle.HIGHEST_PROTOCOL))
                    transaction.delete(*(f"{pool}:{suffix}" for suffix in ("groups", "ready", "claimed", "prepared_ids", "prepared", "done")))
                    transaction.execute()
                learner._synchronize_cuda()
                success = True
                print(f"[Overlap Learner GPU {learner.gpu_id}] step={learner.global_step}, "
                      f"local_rollouts={report['rollouts']}, overlap={overlap_seconds:.3f}s", flush=True)
            except BaseException as error:
                learner.round_gate.abort(str(error))
                raise
            finally:
                try:
                    if learner.cpu_offload:
                        learner._offload_model()
                    else:
                        learner._release_shared_gpu_turn_memory(suppress_errors=not success)
                    learner._synchronize_cuda()
                finally:
                    learner.gpu_gate.release()
                if success:
                    learner.round_gate.learner_done(learner.rank, manifest.round_id)
                    last_round = manifest.round_id
    finally:
        client.close()
