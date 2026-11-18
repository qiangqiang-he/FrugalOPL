"""Paired, first-token head/tail intervention on AMC23 rollouts."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import queue
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from utils.answer_verifier import extract_final_answer, verify_response_answer
from utils.prompts import MATH_PROMPT, render_prompt_from_record

MAX_TOKENS = 8192
ROLLOUTS = 8
TRAJECTORIES = 32
PREFIXES = 10
BRANCHES = ("head", "tail")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose_prefixes(initial, rng):
    selected = []
    for correct in (True, False):
        pool = [row for row in initial if row["correct"] == correct and len(row["token_ids"]) >= PREFIXES]
        if len(pool) < TRAJECTORIES:
            raise ValueError(f"Only {len(pool)} eligible trajectories with correct={correct}; need {TRAJECTORIES}")
        for row in rng.sample(pool, TRAJECTORIES):
            for length in sorted(rng.sample(range(len(row["token_ids"])), PREFIXES)):
                selected.append({
                    "pair": len(selected), "question": row["question"], "rollout": row["rollout"],
                    "original_correct": correct, "prefix_length": length,
                })
    return selected


def prompt_ids(tokenizer, record):
    message = {"role": "user", "content": render_prompt_from_record(MATH_PROMPT, record)}
    rendered = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    return tokenizer.encode(rendered, add_special_tokens=False)


def evaluation_engine(model, *, interventions=True, max_num_seqs=64):
    from transformers import GenerationConfig
    from vllm import LLM

    generation_config = GenerationConfig.from_pretrained(model, local_files_only=True)
    eos = generation_config.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else eos
    options = {}
    if interventions:
        # Chunked prefill can run the first-token processor on partial prompts.
        options.update(logits_processors=["scripts.amc23_head_tail_logits:HeadTailLogitsProcessor"],
                       enable_chunked_prefill=False, async_scheduling=False)
    engine = LLM(
        model=model, dtype="bfloat16",
        generation_config="vllm", max_model_len=MAX_TOKENS + 1024, max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.65, enable_prefix_caching=True,
        max_num_batched_tokens=16384, **options,
    )
    return engine, stop_ids


def gpu_worker(device, args, records, requests, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from vllm import SamplingParams

    engine, stop_ids = evaluation_engine(args.model)
    tokenizer = engine.get_tokenizer()
    questions = [prompt_ids(tokenizer, record) for record in records]
    if max(map(len, questions)) > 1024:
        raise ValueError("A prompt exceeds the context reserved for AMC23 questions")
    while (task := requests.get()) is not None:
        phase, batch, rows = task
        trace_path = args.output.resolve() / f".empty-tails-{batch:03}.jsonl"
        if phase == "five":
            trace_path.unlink(missing_ok=True)
        prompts, sampling = [], []
        for row in rows:
            prefix = row.get("prefix_ids", [])
            prompts.append({"prompt_token_ids": questions[row["question"]] + prefix})
            branch_args = None
            if phase in {"branch", "five"}:
                branch_args = {"pair": row["pair"], "branch": row["branch"]}
            if phase == "five":
                branch_args.update(actions=row["actions"], tail_threshold=row["tail_threshold"],
                                   empty_tail_log=str(trace_path), stop_token=stop_ids[0])
            sampling.append(SamplingParams(
                temperature=1.0, top_p=1.0, top_k=-1, detokenize=False, seed=row["seed"],
                max_tokens=MAX_TOKENS - len(prefix), stop_token_ids=stop_ids, extra_args=branch_args,
            ))
        outputs = engine.generate(prompts, sampling_params=sampling, use_tqdm=True)
        empty_steps = {}
        if phase == "five" and trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["pair"] in empty_steps:
                    raise RuntimeError(f"Duplicate empty-tail event for prefix {event['pair']}")
                empty_steps[event["pair"]] = event["step"]
            trace_path.unlink()
        scored = []
        for row, result in zip(rows, outputs, strict=True):
            output, = result.outputs
            new_ids = list(output.token_ids)
            prefix = row.get("prefix_ids", [])
            if not new_ids or len(prefix) + len(new_ids) > MAX_TOKENS:
                raise RuntimeError(f"Invalid output length for {row['question']}: {len(prefix) + len(new_ids)}")
            empty_step = empty_steps.pop(row["pair"], None) if row.get("branch") == "tail" else None
            if empty_step is not None:
                if len(new_ids) != empty_step or new_ids[-1] != stop_ids[0] or output.finish_reason != "stop":
                    raise RuntimeError(f"The empty tail did not stop at step {empty_step}: {row['pair']}")
                new_ids.pop()
            token_ids = prefix + new_ids
            response = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answer, valid = extract_final_answer(response)
            correct = None if empty_step is not None else bool(
                verify_response_answer(response, str(records[row["question"]]["answer"]))
            )
            saved = {key: value for key, value in row.items() if key != "prefix_ids"}
            saved.update(token_ids=token_ids, response=response, correct=correct,
                         boxed_answer=answer if valid else None, finish_reason=output.finish_reason)
            if phase == "branch":
                saved["intervention_token_id"] = new_ids[0]
            if phase == "five":
                saved["action_token_ids"] = new_ids[:row["actions"]]
                saved["empty_tail_step"] = empty_step
            scored.append(saved)
        if empty_steps:
            raise RuntimeError(f"Unexpected empty-tail events in batch {batch}: {empty_steps}")
        results.put((phase, batch, scored))


def progress(completed, total, stage, started, new_responses, unit="responses"):
    elapsed = time.monotonic() - started
    speed = new_responses / elapsed
    remaining = (total - completed) / speed
    now = datetime.now(timezone.utc)
    print(f"{now:%Y-%m-%dT%H:%M:%SZ} Eval: {completed}/{total} {unit} | {stage} | "
          f"{speed:.2f} {unit}/s | elapsed={elapsed:.0f}s | remaining={remaining:.0f}s", flush=True)
    print("GPU_PROGRESS " + json.dumps({"phase": "eval", "step": completed, "total": total,
          "remaining_seconds": round(remaining), "updated_at": int(now.timestamp())}), flush=True)


def run_phase(phase, rows, output, requests, results, workers, progress_state, *, batch_size=None):
    completed_batches = {}
    expected = {}
    if batch_size is None:
        batch_size = 40 if phase == "initial" else 64
    for batch, start in enumerate(range(0, len(rows), batch_size)):
        chunk = rows[start:start + batch_size]
        path = output / f"{phase}-{batch:03}.json"
        if path.is_file():
            completed_batches[batch] = read_json(path)
            progress_state["completed"] += len(completed_batches[batch])
        else:
            expected[batch] = path
            requests.put((phase, batch, chunk))
    while expected:
        try:
            received_phase, batch, scored = results.get(timeout=5)
        except queue.Empty:
            failed = [worker for worker in workers if worker.exitcode is not None]
            if failed:
                raise RuntimeError(f"GPU workers exited early: {[(w.name, w.exitcode) for w in failed]}")
            continue
        if received_phase != phase or batch not in expected:
            raise RuntimeError(f"Unexpected result for {received_phase} batch {batch}")
        write_json(expected.pop(batch), scored)
        completed_batches[batch] = scored
        progress_state["completed"] += len(scored)
        progress_state["new_responses"] += len(scored)
        progress(stage=phase, **progress_state)
    return [row for batch in sorted(completed_batches) for row in completed_batches[batch]]


def summarize(args, records, initial, selected, branches):
    if len(initial) != len(records) * ROLLOUTS or len(branches) != len(selected) * 2:
        raise RuntimeError("Unexpected number of initial or intervention rollouts")
    if {(row["question"], row["rollout"]) for row in initial} != {
        (question, rollout) for question in range(len(records)) for rollout in range(ROLLOUTS)
    }:
        raise RuntimeError("Missing initial rollout IDs")
    if {(row["pair"], row["branch"]) for row in branches} != {
        (row["pair"], branch) for row in selected for branch in BRANCHES
    }:
        raise RuntimeError("Missing paired continuation IDs")
    metrics = {}
    for correct, label in ((True, "original_correct"), (False, "original_incorrect")):
        metrics[label] = {}
        for branch in BRANCHES:
            group = [row for row in branches if row["original_correct"] == correct and row["branch"] == branch]
            if len(group) != TRAJECTORIES * PREFIXES:
                raise RuntimeError(f"Wrong sample count in {label}/{branch}: {len(group)}")
            successes = sum(row["correct"] for row in group)
            metrics[label][branch] = {"correct": successes, "total": len(group), "accuracy": successes / len(group)}
    return {
        "model": "Qwen3-1.7B", "model_path": args.model, "dataset": "AMC23", "questions": len(records),
        "seed": args.seed, "enable_thinking": False, "rollouts_per_question": ROLLOUTS,
        "max_response_tokens": MAX_TOKENS, "top_p": 1.0, "temperature": 1.0, "head_threshold": 0.9,
        "trajectories_per_group": TRAJECTORIES, "prefixes_per_trajectory": PREFIXES,
        "initial_correct": sum(row["correct"] for row in initial), "initial_total": len(initial),
        "initial_too_short": sum(len(row["token_ids"]) < PREFIXES for row in initial),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = read_json(args.dataset)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if not all(device.isdecimal() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to distinct GPU indices")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, records, devices)


@contextmanager
def gpu_workers(args, records, devices, *, worker_target=gpu_worker):
    context = mp.get_context("spawn")
    requests, results = context.Queue(), context.Queue()
    workers = [context.Process(target=worker_target, name=f"gpu-{device}",
                               args=(device, args, records, requests, results)) for device in devices]
    for worker in workers:
        worker.start()
    try:
        yield requests, results, workers
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            requests.put(None)
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        requests.cancel_join_thread()


def run(args, records, devices):
    rng = random.Random(args.seed)
    initial_requests = [{"question": question, "rollout": rollout, "seed": rng.randrange(2**31)}
                        for question in range(len(records)) for rollout in range(ROLLOUTS)]
    progress_state = {"completed": 0, "total": len(initial_requests) + 4 * TRAJECTORIES * PREFIXES,
                      "started": time.monotonic(), "new_responses": 0}
    print(f"Eval: 0/{progress_state['total']} responses | loading models | ETA pending", flush=True)
    with gpu_workers(args, records, devices) as (requests, results, workers):
        initial = run_phase("initial", initial_requests, args.output, requests, results, workers, progress_state)
        selected = choose_prefixes(initial, rng)
        write_json(args.output / "prefixes.json", selected)
        sources = {(row["question"], row["rollout"]): row["token_ids"] for row in initial}
        branch_requests = [{**row, "branch": branch, "seed": rng.randrange(2**31),
                            "prefix_ids": sources[(row["question"], row["rollout"])][:row["prefix_length"]]}
                           for row in selected for branch in BRANCHES]
        # Mix groups across devices, keeping the two branches of every prefix adjacent.
        pairs = list(range(len(selected)))
        rng.shuffle(pairs)
        branch_requests = [branch_requests[2 * pair + side] for pair in pairs for side in range(2)]
        branches = run_phase("branch", branch_requests, args.output, requests, results, workers, progress_state)
        summary = summarize(args, records, initial, selected, branches)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
