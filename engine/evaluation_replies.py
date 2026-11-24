"""Distribute every evaluation question's replies across Actor GPUs."""
from __future__ import annotations

import copy
import hashlib
import pickle
import multiprocessing as mp


def reply_shards(question_count, replies, gpu_ids):
    if question_count < 1 or replies < 1 or not gpu_ids:
        raise ValueError("Evaluation reply sharding requires questions, replies and GPUs.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Evaluation Actor GPUs must be distinct.")
    queues = [[] for _ in gpu_ids]
    for sample in range(replies):
        for question in range(question_count):
            queues[(question * replies + sample) % len(gpu_ids)].append((question, sample))
    return [(gpu, pairs) for gpu, pairs in zip(gpu_ids, queues) if pairs]


def reply_seed(seed, identity, question, sample):
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}:{question}:{sample}".encode()).digest()[:4], "big")


def merge_reply_trajectories(parts, question_count, replies):
    import torch
    rows, seen = [], set()
    fields = set(parts[0][0])
    for trajectory, pairs in parts:
        if set(trajectory) != fields or any(len(values) != len(pairs) for values in trajectory.values()):
            raise ValueError("Evaluation reply fields must align with their candidate IDs.")
        for index, raw_pair in enumerate(pairs):
            pair = tuple(raw_pair)
            if pair in seen or not (0 <= pair[0] < question_count and 0 <= pair[1] < replies):
                raise ValueError("Duplicate or invalid evaluation reply ID.")
            if int(trajectory["prompt_indices"][index]) != pair[0]:
                raise ValueError("Evaluation prompt index disagrees with its reply ID.")
            seen.add(pair)
            rows.append((pair, trajectory, index))
    if len(seen) != question_count * replies:
        raise ValueError("Evaluation is missing required replies.")
    rows.sort(key=lambda row: row[0])
    result = {}
    for name in fields:
        values = [trajectory[name][index] for _, trajectory, index in rows]
        result[name] = torch.stack(values) if isinstance(parts[0][0][name], torch.Tensor) else values
    return result


def run_distributed_reply_evaluation(controller, client, *, records, evaluation_id, cache,
                                     question_results, stop_from_partial, actor_manager,
                                     learner_manager, teacher_manager):
    from utils.eval import build_question_results
    replies = controller.eval_rollouts_per_prompt
    gpu_ids = tuple(controller.scheduling.actor_gpu_ids)
    cap = min(64, getattr(controller, "eval_rollouts_per_batch", 64))
    wave_size = max(1, cap * len(gpu_ids) // replies)
    if replies > cap * len(gpu_ids):
        raise ValueError("Evaluation replies for one question exceed combined Actor call capacity.")
    remaining = []
    for index, record in enumerate(records):
        saved = cache.load(index, index, 1, replies)
        if saved is None:
            remaining.append((index, record))
        else:
            question_results.extend(saved)
    stop = stop_from_partial()
    if stop:
        return stop
    keys = controller.queue_keys()
    for offset in range(0, len(remaining), wave_size):
        wave = remaining[offset:offset + wave_size]
        wave_records = [record for _, record in wave]
        pending = {}
        for gpu, pairs in reply_shards(len(wave), replies, gpu_ids):
            questions, answers = controller._records_to_signals(wave_records)
            message = dict(evaluation_id=evaluation_id, evaluation_batch_index=offset,
                           evaluation_gpu_id=gpu, policy_version=controller._last_policy_version,
                           questions=questions, answers=answers, records=copy.deepcopy(wave_records),
                           evaluation_reply_plan={"pairs": pairs, "replies": replies,
                               "record_indices": [index for index, _ in wave],
                               "identity": evaluation_id.rsplit("-", 1)[0]})
            client.lpush(f"{keys['actor_eval']}:gpu:{gpu}", pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))
            pending[gpu] = pairs
        parts = []
        while pending:
            result = controller._pop_message(client, result_key=keys["eval_results"],
                actor_manager=actor_manager, learner_manager=learner_manager, teacher_manager=teacher_manager)
            gpu = result.get("evaluation_gpu_id")
            if (isinstance(gpu, bool) or gpu not in pending or result.get("evaluation_id") != evaluation_id
                    or result.get("evaluation_batch_index") != offset
                    or result.get("actor_metrics", {}).get("lora_step") != controller._last_policy_version
                    or result.get("evaluation_reply_plan", {}).get("pairs") != pending[gpu]):
                raise RuntimeError("Evaluation reply result has the wrong GPU, policy, wave or candidate IDs.")
            parts.append((result["traj"], pending.pop(gpu)))
        trajectory = merge_reply_trajectories(parts, len(wave), replies)
        results = build_question_results(wave_records, trajectory, rollouts_per_prompt=replies,
                                         save_rollouts=controller.save_eval_rollouts)
        for (original_index, _), question in zip(wave, results, strict=True):
            question["record_index"] = original_index
            cache.save(original_index, original_index, [question])
            question_results.append(question)
        print(f"[Controller] Distributed evaluation replies: {len(question_results)*replies}/{len(records)*replies}.", flush=True)
        stop = stop_from_partial()
        if stop:
            return stop
    return None


def run_offline_reply_pool(config, tasks):
    """Use directed Actor queues for the standalone start_eval.sh entry point."""
    import torch
    from utils.eval import (_evaluation_worker, _worker_message_or_raise, _stop_workers,
                            _output_directory, build_question_results, write_json)
    gpu_ids = config["actor"]["gpu_ids"]
    if (not gpu_ids or len(set(gpu_ids)) != len(gpu_ids)
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids)):
        raise ValueError("Offline reply evaluation requires distinct Actor GPUs.")
    options = config["actor"]["evaluation"]
    replies = options["rollouts_per_prompt"]
    cap = min(64, options.get("rollouts_per_batch", 64))
    context = mp.get_context(config["run"].get("runtime", {}).get("process_start_method", "spawn"))
    queues = {gpu: context.Queue() for gpu in gpu_ids}
    results_queue = context.Queue()
    workers = [context.Process(target=_evaluation_worker, args=(gpu, config, queues[gpu], results_queue),
                              name=f"FrugalOPL-Eval-Replies-GPU{gpu}") for gpu in gpu_ids]
    for worker in workers:
        worker.start()
    completed, force, subtask = {}, True, 0
    try:
        ready = set()
        while len(ready) < len(workers):
            message = _worker_message_or_raise(results_queue, workers)
            if message.get("kind") != "ready" or message.get("gpu_id") not in queues:
                raise RuntimeError("Invalid offline reply Actor readiness message.")
            ready.add(message["gpu_id"])
        offset = 0
        while offset < len(tasks):
            wave, count = [], 0
            dataset, alias = tasks[offset]["dataset_name"], tasks[offset]["prompt_alias"]
            while offset < len(tasks):
                task = tasks[offset]
                if (task["dataset_name"] != dataset or task["prompt_alias"] != alias
                        or count + len(task["records"]) > cap * len(gpu_ids) // replies):
                    break
                wave.append(task)
                count += len(task["records"])
                offset += 1
            if not wave:
                raise ValueError("An offline evaluation task exceeds combined Actor call capacity.")
            records = [record for task in wave for record in task["records"]]
            record_indices = [task["record_offset"] + i for task in wave for i in range(len(task["records"]))]
            pending = {}
            for gpu, pairs in reply_shards(len(records), replies, tuple(gpu_ids)):
                plan = {"pairs": pairs, "replies": replies, "record_indices": record_indices,
                        "identity": f"offline-{config['run'].get('name', '')}-{dataset}-{alias}"}
                queues[gpu].put({"task_id": subtask, "records": records, "prompt_alias": alias, "reply_plan": plan})
                pending[subtask] = (gpu, pairs)
                subtask += 1
            parts = []
            while pending:
                message = _worker_message_or_raise(results_queue, workers)
                identifier = message.get("task_id")
                if message.get("kind") != "result" or identifier not in pending:
                    raise RuntimeError("Invalid or duplicate offline evaluation reply result.")
                gpu, pairs = pending.pop(identifier)
                if message.get("gpu_id") != gpu:
                    raise RuntimeError("Offline reply result came from the wrong GPU.")
                parts.append((message["actor_result"]["traj"], pairs))
            merged = merge_reply_trajectories(parts, len(records), replies)
            question_offset = 0
            for task in wave:
                first, stop = question_offset * replies, (question_offset + len(task["records"])) * replies
                trajectory = {key: value[first:stop] for key, value in merged.items()}
                trajectory["prompt_indices"] = torch.as_tensor(trajectory["prompt_indices"]) - question_offset
                questions = build_question_results(task["records"], trajectory, rollouts_per_prompt=replies,
                    save_rollouts=options.get("save_rollouts", False), start_index=task["record_offset"])
                for q, question in enumerate(questions):
                    for sample, rollout in enumerate(question["rollouts"]):
                        rollout["actor_gpu_id"] = gpu_ids[((question_offset + q) * replies + sample) % len(gpu_ids)]
                identifier = task["task_id"]
                completed[identifier] = {"question_results": questions}
                write_json(_output_directory(config, config_path=None) / "rollouts" / f"batch_{identifier:05d}.json",
                    {"task_id": identifier, "dataset": dataset, "reply_layout": "striped_replies_v1", "questions": questions})
                question_offset += len(task["records"])
            print(f"[Evaluation] Distributed replies completed {len(completed)}/{len(tasks)} batches; dataset={dataset}.", flush=True)
        force = False
        return completed
    finally:
        # Signal all workers first; then join independently directed queues.
        for queue in queues.values():
            queue.put(None)
        for worker, queue in zip(workers, queues.values()):
            _stop_workers([worker], queue, force=force)
            queue.close()
        results_queue.close()
