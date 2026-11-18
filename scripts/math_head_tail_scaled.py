"""Run the single-token head/tail experiment across AMC23, MATH500, and AIME24."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from scripts.amc23_head_tail import BRANCHES, MAX_TOKENS, gpu_workers, read_json, run_phase, write_json
from scripts.amc23_head_tail_filtered import (
    HEAD_THRESHOLD, PREFIXES, TAIL_THRESHOLD, rollout_worker, screen_worker, summarize,
)

DATASETS = {
    "AMC23": ("AMC-2023.json", 40, 107, 8),
    "MATH500": ("MATH500.json", 500, 107, 8),
    "AIME24": ("AIME-2024.json", 30, 106, 32),
}
EXTRA_ROLLOUTS_PER_QUESTION = {"AMC23": 8, "MATH500": 4, "AIME24": 16}
GENERATION_BATCH_SIZE = 256


def datasets(root):
    records, indices = [], {}
    for name, (filename, size, _, _) in DATASETS.items():
        questions = read_json(root / filename)
        if len(questions) != size:
            raise ValueError(f"Expected {size} questions in {name}, got {len(questions)}")
        indices[name] = tuple(range(len(records), len(records) + len(questions)))
        records.extend({**row, "dataset": name} for row in questions)
    return records, indices


def seed_for(seed, phase, question, replicate):
    return random.Random(f"{seed}/{phase}/{question}/{replicate}").randrange(2**31)


def original_amc23(prior_output, question_ids):
    rows = [r for batch in sorted(prior_output.glob("initial-*.json")) for r in read_json(batch)]
    original = {(r["question"], r["rollout"]): r for r in rows}
    expected = {(q, r) for q in question_ids for r in range(DATASETS["AMC23"][3])}
    if len(rows) != len(expected) or set(original) != expected:
        raise ValueError("The saved AMC23 initial rollout pool is incomplete")
    return original


def initial_requests(indices, starts, counts, seed):
    rows = [{"question": question, "rollout": replicate, "prefix_ids": [],
             "seed": seed_for(seed, "initial", question, replicate)}
            for dataset, count in counts.items() for question in indices[dataset]
            for replicate in range(starts[dataset], starts[dataset] + count)]
    random.Random(seed_for(seed, "initial_order", len(rows), sum(starts.values()))).shuffle(rows)
    return rows


def progress_state(total, *, unit="responses"):
    return {"completed": 0, "total": total, "started": time.monotonic(), "new_responses": 0, "unit": unit}


def collect_initial(args, phase, indices, starts, counts, original, inference):
    rows = initial_requests(indices, starts, counts, args.seed)
    print(f"Eval: 0/{len(rows)} responses | {phase} | waiting for rollouts", flush=True)
    results = run_phase(phase, rows, args.output, *inference, progress_state(len(rows)),
                        batch_size=GENERATION_BATCH_SIZE)
    for row in results:
        key = row["question"], row["rollout"]
        if key in original or not row["token_ids"]:
            raise RuntimeError(f"Duplicate or empty initial rollout {key}")
        original[key] = row
    for name, count in counts.items():
        starts[name] += count


def screen_candidates(records, original, visited, accepted, seed):
    groups = {key: [] for key in accepted}
    for key, row in original.items():
        group = records[row["question"]]["dataset"], row["correct"]
        if key not in visited and len(accepted[group]) < DATASETS[group[0]][2]:
            groups[group].append(row)
    candidates = []
    for group, pool in groups.items():
        needed = DATASETS[group[0]][2] - len(accepted[group])
        pool.sort(key=lambda r: (seed_for(seed, "cohort", r["question"], r["rollout"]),
                                 r["question"], r["rollout"]))
        candidates.extend({**row, "seed": seed_for(seed, "screen", row["question"], row["rollout"])}
                          for row in pool[:needed])
    return candidates


def build_cohorts(args, records, indices, original, starts, devices, inference):
    accepted = {(name, correct): [] for name in DATASETS for correct in (True, False)}
    visited, screened = set(), []
    screen_wave, extension = 0, 0
    progress = progress_state(0, unit="trajectories")
    with gpu_workers(args, records, devices, worker_target=screen_worker) as screening:
        while True:
            missing = {name for (name, _), group in accepted.items() if len(group) < DATASETS[name][2]}
            if not missing:
                break
            candidates = screen_candidates(records, original, visited, accepted, args.seed)
            if not candidates:
                extra = {name: EXTRA_ROLLOUTS_PER_QUESTION[name] for name in DATASETS if name in missing}
                print("Supplementing initial trajectories " + json.dumps(extra), flush=True)
                collect_initial(args, f"extra-{extension:02}", indices, starts, extra, original, inference)
                extension += 1
                continue
            progress["total"] = progress["completed"] + len(candidates)
            scanned = run_phase(f"screen-{screen_wave:02}", candidates, args.output, *screening, progress, batch_size=8)
            screen_wave += 1
            for row in scanned:
                key = row["question"], row["rollout"]
                visited.add(key)
                name = records[row["question"]]["dataset"]
                screened.append({k: v for k, v in row.items() if k != "prefixes"} |
                                {"dataset": name, "eligible_found": len(row["prefixes"])})
                if len(row["prefixes"]) == PREFIXES:
                    accepted[name, row["correct"]].append(row)
            print("Eligible sources " + json.dumps({f"{name}/{correct}": len(group)
                  for (name, correct), group in accepted.items()}), flush=True)

    plan = []
    for (name, correct), group in accepted.items():
        for row in group:
            for prefix in row["prefixes"]:
                plan.append({"pair": len(plan), "dataset": name, "question": row["question"],
                             "rollout": row["rollout"], "original_correct": correct, **prefix})
    return plan, screened


def continuation_requests(plan, original, seed):
    shuffled = list(plan)
    random.Random(seed).shuffle(shuffled)
    rows = []
    for prefix in shuffled:
        source = original[(prefix["question"], prefix["rollout"])]["token_ids"][:prefix["prefix_length"]]
        common = {key: prefix[key] for key in (
            "pair", "dataset", "question", "rollout", "original_correct", "prefix_length",
        )}
        for i, branch in enumerate(BRANCHES):
            token = prefix[branch]["id"]
            rows.append({**common, "branch": branch, "action_token_id": token, "prefix_ids": source + [token],
                         "seed": seed_for(seed, "continuation", prefix["pair"], i)})
    return rows


def summarize_datasets(args, plan, outputs, original, reused, screened, starts):
    expected_trajectories_per_label = sum(spec[2] for spec in DATASETS.values())
    if len(outputs) != 4 * expected_trajectories_per_label * PREFIXES:
        raise RuntimeError("The total is smaller or larger than the planned tenfold evaluation")
    result = {
        "model": "Qwen3-1.7B", "model_path": args.model, "seed": args.seed, "enable_thinking": False,
        "head_threshold": HEAD_THRESHOLD, "tail_upper_cumulative": TAIL_THRESHOLD,
        "max_next_token_probability": HEAD_THRESHOLD, "actions": 1, "max_response_tokens": MAX_TOKENS,
        "top_p_after_action": 1.0, "temperature": 1.0, "prefixes_per_trajectory": PREFIXES,
        "trajectories_per_label": expected_trajectories_per_label, "prefix_count": len(plan),
        "response_count": len(outputs), "datasets": {}, "overall": {},
    }
    metadata = {r["question"]: r["dataset"] for r in args.records_by_question}
    for name, (_, questions, quota, _) in DATASETS.items():
        prefixes = [p for p in plan if p["dataset"] == name]
        responses = [r for r in outputs if r["dataset"] == name]
        if any(metadata[p["question"]] != name for p in prefixes + responses):
            raise RuntimeError(f"Inconsistent dataset IDs in {name}")
        checks = summarize(args, prefixes, responses, original, reused,
                           [r for r in screened if r["dataset"] == name], trajectories_per_group=quota)
        pool = [r for r in original.values() if metadata[r["question"]] == name]
        result["datasets"][name] = {
            "questions": questions, "initial_rollouts_per_question": starts[name], "initial_total": len(pool),
            "initial_correct": sum(r["correct"] for r in pool), "trajectories_per_label": quota,
            "metrics": checks["metrics"],
        }
    for label in ("original_correct", "original_incorrect"):
        result["overall"][label] = {}
        for branch in BRANCHES:
            results = [d["metrics"][label][branch] for d in result["datasets"].values()]
            total, correct = (sum(row[key] for row in results) for key in ("total", "correct"))
            result["overall"][label][branch] = {"total": total, "correct": correct, "accuracy": correct / total}
    return result


def run(args, devices):
    records, indices = datasets(args.dataset_root)
    args.records_by_question = [{"question": i, "dataset": row["dataset"]} for i, row in enumerate(records)]
    original = original_amc23(args.prior_output, indices["AMC23"])
    reused = set(original)
    starts = {name: 0 for name in DATASETS}
    starts["AMC23"] = DATASETS["AMC23"][3]
    counts = {name: DATASETS[name][3] for name in ("MATH500", "AIME24")}
    with gpu_workers(args, records, devices, worker_target=rollout_worker) as inference:
        collect_initial(args, "initial-new", indices, starts, counts, original, inference)
        plan, screened = build_cohorts(args, records, indices, original, starts, devices, inference)
        write_json(args.output / "prefixes.json", plan)
        write_json(args.output / "screening.json", screened)
        requests = continuation_requests(plan, original, args.seed)
        print(f"Eval: 0/{len(requests)} responses | starting paired continuations", flush=True)
        outputs = run_phase("continuation", requests, args.output, *inference, progress_state(len(requests)),
                            batch_size=GENERATION_BATCH_SIZE)
    summary = summarize_datasets(args, plan, outputs, original, reused, screened, starts)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(generation_slots=128)
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
