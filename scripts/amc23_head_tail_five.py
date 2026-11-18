"""Compare five consecutive head actions with five consecutive tail actions on AMC23."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from scripts.amc23_head_tail import (
    BRANCHES, MAX_TOKENS, TRAJECTORIES, gpu_workers, read_json, run_phase, write_json,
)

ACTIONS = 5
PREFIXES = 32
TAIL_THRESHOLD = 0.99


def source_trajectories(prior_output):
    initial = {(row["question"], row["rollout"]): row
               for batch in sorted(prior_output.glob("initial-*.json")) for row in read_json(batch)}
    old_plan = read_json(prior_output / "prefixes.json")
    keys = sorted({(row["question"], row["rollout"]) for row in old_plan})
    sources = {key: initial[key] for key in keys}
    if Counter(row["correct"] for row in sources.values()) != {True: TRAJECTORIES, False: TRAJECTORIES}:
        raise ValueError("Expected 32 correct and 32 incorrect source trajectories from the previous experiment")
    return sources


def choose_prefixes(sources, rng):
    plan = []
    for (question, rollout), row in sources.items():
        # Offsets precede an existing token and leave room for all five actions.
        limit = min(len(row["token_ids"]), MAX_TOKENS - ACTIONS + 1)
        if limit < PREFIXES:
            raise ValueError(f"Source trajectory {question}/{rollout} has only {limit} eligible offsets")
        for length in sorted(rng.sample(range(limit), PREFIXES)):
            plan.append({"pair": len(plan), "question": question, "rollout": rollout,
                         "original_correct": row["correct"], "prefix_length": length})
    return plan


def score(rows):
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"correct": correct, "total": count, "accuracy": correct / count if count else None,
            "early_eos": sum(len(row["action_token_ids"]) < ACTIONS for row in rows)}


def summarize(args, plan, branches, sources):
    results = {(row["pair"], row["branch"]): row for row in branches}
    expected = {(prefix["pair"], branch) for prefix in plan for branch in BRANCHES}
    if len(branches) != len(expected) or set(results) != expected:
        raise RuntimeError("Missing or duplicate branch results")
    metrics = {}
    for original_correct, label in ((True, "original_correct"), (False, "original_incorrect")):
        all_head, valid_head, valid_tail = [], [], []
        empty_step_counts = {str(step): 0 for step in range(1, ACTIONS + 1)}
        for prefix in plan:
            if prefix["original_correct"] != original_correct:
                continue
            source = sources[(prefix["question"], prefix["rollout"])]["token_ids"]
            length = prefix["prefix_length"]
            head, tail = (results[prefix["pair"], name] for name in BRANCHES)
            for row in (head, tail):
                if any(row[key] != value for key, value in prefix.items()):
                    raise RuntimeError(f"Incorrect prefix metadata for pair {prefix['pair']}")
                tokens, actions = row["token_ids"], row["action_token_ids"]
                if tokens[:length] != source[:length] or actions != tokens[length:length + ACTIONS]:
                    raise RuntimeError(f"Incorrect prefix or action IDs for pair {prefix['pair']}")
                if row["empty_tail_step"] is None and not 0 < len(actions) <= ACTIONS:
                    raise RuntimeError(f"Invalid action sequence for pair {prefix['pair']}")
                if row["empty_tail_step"] is None and len(actions) < ACTIONS and row["finish_reason"] != "stop":
                    raise RuntimeError(f"Short branch without EOS for pair {prefix['pair']}")
            if head["empty_tail_step"] is not None or type(head["correct"]) is not bool:
                raise RuntimeError(f"Invalid head branch for pair {prefix['pair']}")
            if tail["action_token_ids"] and head["action_token_ids"][0] == tail["action_token_ids"][0]:
                raise RuntimeError(f"Head and tail overlap for pair {prefix['pair']}")
            all_head.append(head)
            if tail["empty_tail_step"] is None:
                if type(tail["correct"]) is not bool:
                    raise RuntimeError(f"Missing tail reward for pair {prefix['pair']}")
                valid_head.append(head)
                valid_tail.append(tail)
            else:
                step = tail["empty_tail_step"]
                if tail["correct"] is not None or len(tail["action_token_ids"]) != step - 1:
                    raise RuntimeError(f"Invalid empty-tail record for pair {prefix['pair']}")
                empty_step_counts[str(step)] += 1
        if len(all_head) != TRAJECTORIES * PREFIXES:
            raise RuntimeError(f"Incorrect planned sample count for {label}")
        metrics[label] = {
            "planned_pairs": len(all_head), "valid_pairs": len(valid_tail),
            "coverage": len(valid_tail) / len(all_head), "empty_tail_by_step": empty_step_counts,
            "head": score(valid_head), "tail": score(valid_tail), "head_all": score(all_head),
            "represented_trajectories": len({(r["question"], r["rollout"]) for r in valid_tail}),
            "full_five_action_pairs": sum(len(a["action_token_ids"]) == len(b["action_token_ids"]) == ACTIONS
                                          for a, b in zip(valid_head, valid_tail, strict=True)),
        }
    return {"model": "Qwen3-1.7B", "model_path": args.model, "dataset": "AMC23", "seed": args.seed,
            "enable_thinking": False, "max_response_tokens": MAX_TOKENS, "top_p_after_actions": 1.0,
            "temperature": 1.0, "head_threshold": 0.9, "tail_threshold": TAIL_THRESHOLD,
            "actions_per_branch": ACTIONS, "trajectories_per_group": TRAJECTORIES,
            "prefixes_per_trajectory": PREFIXES, "initial_source": str(args.prior_output), "metrics": metrics}


def run(args, devices):
    records = read_json(args.dataset)
    sources = source_trajectories(args.prior_output)
    rng = random.Random(args.seed)
    plan = choose_prefixes(sources, rng)
    path = args.output / "prefixes.json"
    if path.is_file() and read_json(path) != plan:
        raise ValueError("The saved prefix plan differs from the requested run")
    write_json(path, plan)

    shuffled = list(plan)
    rng.shuffle(shuffled)
    requests = [{**prefix, "branch": branch, "seed": rng.randrange(2**31), "actions": ACTIONS,
                 "tail_threshold": TAIL_THRESHOLD,
                 "prefix_ids": sources[(prefix["question"], prefix["rollout"])]["token_ids"][:prefix["prefix_length"]]}
                for prefix in shuffled for branch in BRANCHES]
    progress = {"completed": 0, "total": len(requests), "started": time.monotonic(),
                "new_responses": 0, "unit": "branch attempts"}
    print(f"Eval: 0/{len(requests)} branch attempts | loading models | ETA pending", flush=True)
    with gpu_workers(args, records, devices) as (work_queue, results, workers):
        branches = run_phase("five", requests, args.output, work_queue, results, workers, progress)
        summary = summarize(args, plan, branches, sources)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if not all(device.isdecimal() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to distinct GPU indices")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, devices)


if __name__ == "__main__":
    main()
