"""Single-action AMC23 head/tail evaluation with conditional prefix screening."""

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
    BRANCHES, MAX_TOKENS, TRAJECTORIES, evaluation_engine, gpu_workers, prompt_ids,
    read_json, run_phase, write_json,
)
from utils.answer_verifier import extract_final_answer, verify_response_answer

PREFIXES = 32
HEAD_THRESHOLD = 0.9
TAIL_THRESHOLD = 0.999
LOGIT_BATCH_SIZE = 128


def draw_actions(probabilities, generator):
    """Sample both actions from one distribution for each eligible position."""
    import torch

    eligible = (probabilities.amax(dim=1) <= HEAD_THRESHOLD).nonzero().flatten()
    high_top1 = len(probabilities) - len(eligible)
    if not len(eligible):
        return [], high_top1, 0
    sorted_probs, token_ids = probabilities[eligible].sort(dim=1, descending=True, stable=True)
    cumulative = sorted_probs.cumsum(dim=1)
    before = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1)
    head = before <= HEAD_THRESHOLD
    tail = (~head) & (cumulative <= TAIL_THRESHOLD) & (sorted_probs > 0)
    usable = tail.any(dim=1)
    empty_tail = len(eligible) - int(usable.sum())
    if not bool(usable.any()):
        return [], high_top1, empty_tail
    eligible, sorted_probs, token_ids = eligible[usable], sorted_probs[usable], token_ids[usable]
    before, cumulative, head, tail = before[usable], cumulative[usable], head[usable], tail[usable]
    head_mass = (sorted_probs * head).sum(dim=1).tolist()
    tail_mass = (sorted_probs * tail).sum(dim=1).tolist()
    top1 = sorted_probs[:, 0].tolist()
    actions = []
    for mask in (head, tail):
        rank = torch.multinomial(sorted_probs * mask, 1, generator=generator)
        action_columns = [tensor.gather(1, rank).flatten().tolist()
                          for tensor in (token_ids, sorted_probs, before, cumulative)]
        actions.append([{"id": token, "probability": probability, "rank": place + 1,
                         "cumulative_before": previous_mass, "cumulative": mass}
                        for token, probability, previous_mass, mass, place in
                        zip(*action_columns, rank.flatten().tolist(), strict=True)])
    return [dict(position=offset, top1_probability=p, head_mass=h, tail_mass=t,
                 head=head_action, tail=tail_action)
            for offset, p, h, t, head_action, tail_action in
            zip(eligible.tolist(), top1, head_mass, tail_mass, *actions, strict=True)], high_top1, empty_tail


def screen_trajectory(model, question_tokens, row):
    import torch

    response_tokens = row["token_ids"]
    offsets = list(range(min(len(response_tokens), MAX_TOKENS)))
    random.Random(row["seed"]).shuffle(offsets)
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(row["seed"])
    # A causal forward over the saved reply supplies the same conditional
    # state as a prefix-only forward at each position, without repeated prefill.
    with torch.inference_mode():
        ids = torch.tensor([question_tokens + response_tokens], device=device)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        selected, screened, high_top1, empty_tail = [], 0, 0, 0
        for start in range(0, len(offsets), LOGIT_BATCH_SIZE):
            batch = offsets[start:start + LOGIT_BATCH_SIZE]
            positions = torch.tensor(batch, device=device) + len(question_tokens) - 1
            probabilities = model.lm_head(hidden.index_select(0, positions)).softmax(dim=1, dtype=torch.float64)
            candidates, high, empty = draw_actions(probabilities, rng)
            for candidate in candidates:
                candidate["prefix_length"] = batch[candidate.pop("position")]
                selected.append(candidate)
            screened += len(batch)
            high_top1 += high
            empty_tail += empty
            if len(selected) >= PREFIXES:
                break
    return {"question": row["question"], "rollout": row["rollout"], "correct": row["correct"],
            "screened": screened, "high_top1": high_top1, "empty_tail": empty_tail,
            "prefixes": selected[:PREFIXES]}


def screen_worker(device, args, records, requests, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda").eval()
    questions = [prompt_ids(tokenizer, record) for record in records]
    print(f"Screening worker ready on GPU {device}, pid={os.getpid()}", flush=True)
    while (task := requests.get()) is not None:
        phase, batch, rows = task
        results.put((phase, batch, [screen_trajectory(model, questions[row["question"]], row) for row in rows]))


def rollout_worker(device, args, records, requests, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from vllm import SamplingParams

    engine, stop_ids = evaluation_engine(args.model, interventions=False, max_num_seqs=getattr(args, "generation_slots", 64))
    tokenizer = engine.get_tokenizer()
    questions = [prompt_ids(tokenizer, record) for record in records]
    if max(map(len, questions)) > 1024:
        raise ValueError("An AMC23 prompt exceeds the context reserved for questions")
    while (task := requests.get()) is not None:
        phase, batch, rows = task
        prompts, sampling = [], []
        for row in rows:
            prefix = row["prefix_ids"]
            if (prefix and prefix[-1] in stop_ids) or len(prefix) == MAX_TOKENS:
                continue
            prompts.append({"prompt_token_ids": questions[row["question"]] + prefix})
            sampling.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, detokenize=False,
                                           seed=row["seed"], max_tokens=MAX_TOKENS - len(prefix),
                                           stop_token_ids=stop_ids))
        outputs = iter(engine.generate(prompts, sampling_params=sampling, use_tqdm=True) if prompts else [])
        scored = []
        for row in rows:
            prefix = row["prefix_ids"]
            token_ids = prefix
            if prefix and prefix[-1] in stop_ids:
                finish_reason = "stop"
            elif len(prefix) == MAX_TOKENS:
                finish_reason = "length"
            else:
                output, = next(outputs).outputs
                token_ids = prefix + list(output.token_ids)
                finish_reason = output.finish_reason
            response = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answer, valid = extract_final_answer(response)
            correct = bool(verify_response_answer(response, str(records[row["question"]]["answer"])))
            saved = {key: value for key, value in row.items() if key != "prefix_ids"}
            saved.update(token_ids=token_ids, response=response, correct=correct, finish_reason=finish_reason,
                         boxed_answer=answer if valid else None)
            scored.append(saved)
        results.put((phase, batch, scored))


def trajectory_pools(prior_output, rng):
    original = {(row["question"], row["rollout"]): row
                for batch in sorted(prior_output.glob("initial-*.json")) for row in read_json(batch)}
    if len(original) != 320:
        raise ValueError("Expected 320 original AMC23 rollouts")
    preferred = {(row["question"], row["rollout"]) for row in read_json(prior_output / "prefixes.json")}
    pools = {}
    for correct in (True, False):
        keys = [key for key, row in sorted(original.items()) if row["correct"] == correct]
        priority = [key for key in keys if key in preferred]
        reserves = [key for key in keys if key not in preferred]
        if len(priority) != TRAJECTORIES:
            raise ValueError(f"Expected 32 previously sampled trajectories with correct={correct}")
        rng.shuffle(priority)
        rng.shuffle(reserves)
        pools[correct] = [{**original[key], "seed": rng.randrange(2**31)} for key in priority + reserves]
    return original, preferred, pools


def screen_prefixes(args, records, devices, pools):
    accepted = {True: [], False: []}
    examined = []
    cursors = {True: 0, False: 0}
    progress = {"completed": 0, "new_responses": 0, "started": time.monotonic(), "unit": "trajectories"}
    print(f"Eval: 0/{2 * TRAJECTORIES} trajectories | starting prefix screening | ETA pending", flush=True)
    with gpu_workers(args, records, devices, worker_target=screen_worker) as (requests, results, workers):
        wave = 0
        while any(len(group) < TRAJECTORIES for group in accepted.values()):
            candidates = []
            for correct, group in accepted.items():
                needed = TRAJECTORIES - len(group)
                start = cursors[correct]
                candidates.extend(pools[correct][start:start + needed])
                cursors[correct] = min(start + needed, len(pools[correct]))
            if not candidates:
                raise RuntimeError(f"Insufficient eligible trajectories: {[(k, len(v)) for k, v in accepted.items()]}")
            progress["total"] = progress["completed"] + len(candidates)
            scanned = run_phase(f"screen-{wave:02}", candidates, args.output, requests, results, workers,
                                progress, batch_size=4)
            examined.extend(scanned)
            for row in scanned:
                if len(row["prefixes"]) == PREFIXES:
                    accepted[row["correct"]].append(row)
            wave += 1

    plan = []
    for group in accepted.values():
        for row in group:
            for prefix in row["prefixes"]:
                plan.append({"pair": len(plan), "question": row["question"], "rollout": row["rollout"],
                             "original_correct": row["correct"], **prefix})
    return plan, examined


def summarize(args, plan, outputs, original, preferred, examined, *, trajectories_per_group=TRAJECTORIES):
    pairs = {p["pair"]: p for p in plan}
    if len(outputs) != len(pairs) * 2 or {(r["pair"], r["branch"]) for r in outputs} != {
        (p["pair"], branch) for p in plan for branch in BRANCHES
    }:
        raise RuntimeError("Missing or duplicate paired continuations")
    for row in outputs:
        p = pairs[row["pair"]]
        if any(row[key] != p[key] for key in ("question", "rollout", "original_correct", "prefix_length")):
            raise RuntimeError(f"Incorrect prefix metadata for pair {p['pair']}")
        end = p["prefix_length"]
        source = original[(p["question"], p["rollout"])]["token_ids"]
        action = p[row["branch"]]
        if row["token_ids"][:end] != source[:end] or row["token_ids"][end] != action["id"]:
            raise RuntimeError(f"Prefix or intervention changed during continuation: pair {p['pair']}")
        if row["action_token_id"] != action["id"] or len(row["token_ids"]) > MAX_TOKENS:
            raise RuntimeError(f"Invalid action or response budget: pair {p['pair']}")
    metrics = {}
    for correct, label in ((True, "original_correct"), (False, "original_incorrect")):
        chosen = [p for p in plan if p["original_correct"] == correct]
        counts = Counter((p["question"], p["rollout"]) for p in chosen)
        if len(counts) != trajectories_per_group or set(counts.values()) != {PREFIXES}:
            raise RuntimeError(f"Incorrect trajectory or prefix count for {label}")
        if len(chosen) != len({(p["question"], p["rollout"], p["prefix_length"]) for p in chosen}):
            raise RuntimeError(f"Duplicate prefix positions for {label}")
        for p in chosen:
            head, tail = p["head"], p["tail"]
            if not (p["top1_probability"] <= HEAD_THRESHOLD and head["cumulative_before"] <= HEAD_THRESHOLD
                    < tail["cumulative_before"] <= tail["cumulative"] <= TAIL_THRESHOLD
                    and head["id"] != tail["id"] and p["tail_mass"] > 0):
                raise RuntimeError(f"Out-of-band action sampled for {label}, pair {p['pair']}")
        metrics[label] = {"trajectories_reused": len(preferred.intersection(counts)),
                          "trajectories_screened": sum(r["correct"] == correct for r in examined),
                          "max_top1_probability": max(p["top1_probability"] for p in chosen)}
        for branch in BRANCHES:
            group = [r for r in outputs if r["original_correct"] == correct and r["branch"] == branch]
            if len(group) != len(chosen):
                raise RuntimeError(f"Incorrect output count for {label}/{branch}")
            successes = sum(r["correct"] for r in group)
            metrics[label][branch] = {"correct": successes, "total": len(group), "accuracy": successes / len(group)}
    return {"model": "Qwen3-1.7B", "dataset": "AMC23", "model_path": args.model, "seed": args.seed,
            "enable_thinking": False, "head_threshold": HEAD_THRESHOLD, "tail_upper_cumulative": TAIL_THRESHOLD,
            "max_next_token_probability": HEAD_THRESHOLD, "actions": 1, "max_response_tokens": MAX_TOKENS,
            "top_p_after_action": 1.0, "temperature": 1.0, "trajectories_per_group": trajectories_per_group,
            "prefixes_per_trajectory": PREFIXES, "metrics": metrics}


def run(args, devices):
    records = read_json(args.dataset)
    rng = random.Random(args.seed)
    original, preferred, pools = trajectory_pools(args.prior_output, rng)
    plan_path = args.output / "prefixes.json"
    examined_path = args.output / "screening.json"
    if plan_path.is_file() and examined_path.is_file():
        plan, examined = read_json(plan_path), read_json(examined_path)
    else:
        plan, examined = screen_prefixes(args, records, devices, pools)
        write_json(examined_path, [{k: v for k, v in row.items() if k != "prefixes"} |
                                   {"eligible_found": len(row["prefixes"])} for row in examined])
        write_json(plan_path, plan)

    shuffled = list(plan)
    rng.shuffle(shuffled)
    requests = []
    for p in shuffled:
        prefix = original[(p["question"], p["rollout"])]["token_ids"][:p["prefix_length"]]
        for branch in BRANCHES:
            token = p[branch]["id"]
            requests.append({**{key: p[key] for key in ("pair", "question", "rollout", "original_correct", "prefix_length")},
                             "branch": branch, "action_token_id": token, "prefix_ids": prefix + [token],
                             "seed": rng.randrange(2**31)})
    progress = {"completed": 0, "total": len(requests), "started": time.monotonic(), "new_responses": 0}
    print(f"Eval: 0/{len(requests)} responses | initializing continuation engines | ETA pending", flush=True)
    with gpu_workers(args, records, devices, worker_target=rollout_worker) as (work_queue, out_queue, workers):
        outputs = run_phase("filtered", requests, args.output, work_queue, out_queue, workers, progress)
    summary = summarize(args, plan, outputs, original, preferred, examined)
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
