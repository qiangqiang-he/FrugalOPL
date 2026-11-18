"""Continuous inference-only GRPO-style rollouts and sampled-rank statistics.

One independent, persistent vLLM engine per explicitly selected GPU. Dataset
epochs repeat with fresh deterministic seeds. No optimizer, LoRA or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from utils.prompts import MATH_PROMPT
from utils.sampled_head_tail import METRICS, QUANTILES, StreamingStats, atomic_json


def log(message):
    print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), message, flush=True)


def encode_prompt(tokenizer, question, max_prompt_tokens):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT.format(question=question)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(prompt, Mapping):
        prompt = prompt["input_ids"]
    if not isinstance(prompt, list) or not prompt or not all(isinstance(t, int) for t in prompt):
        raise TypeError("Expected a flat, nonempty list of prompt token IDs")
    return prompt[-max_prompt_tokens:], len(prompt)


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name in ("model", "dataset", "output", "python"):
        if not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be absolute")
    if config["temperature"] != 1. or config["top_p"] != 1. or config["enable_thinking"]:
        raise ValueError("This study requires T=1, top_p=1 and no thinking")
    if not 0 < config["max_tokens"] <= 8192:
        raise ValueError("max_tokens must be in [1,8192]")
    if config["questions_per_batch"] % len(config["gpus"]):
        raise ValueError("questions_per_batch must divide evenly across GPUs")
    return config


def fingerprint(config):
    model = Path(config["model"])
    dataset_bytes = Path(config["dataset"]).read_bytes()
    rows = json.loads(dataset_bytes)
    if not rows or any(not isinstance(r.get("question"), str) for r in rows):
        raise ValueError("Expected a nonempty list of question records")
    if config.get("expected_questions") and len(rows) != config["expected_questions"]:
        raise ValueError(f"Dataset size {len(rows)} != {config['expected_questions']}")
    if not (model / "tokenizer_config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError("Missing local model files (downloads are disabled)")
    model_config = json.loads((model / "config.json").read_text())
    identity = {"config": config,
                "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                "questions": len(rows), "vocab_size": model_config["vocab_size"],
                "partition": "sampled token plus strictly greater logits; other ties are tail",
                "metric_columns": list(METRICS), "observation_columns": ["request_row", "position_1based", "token_id", *METRICS]}
    identity["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, rows


def check_runtime():
    import vllm
    if vllm.__version__ != "0.19.1":
        raise RuntimeError(f"Instrumented vLLM version must be 0.19.1, got {vllm.__version__}")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    for method in ("_sample", "sample_tokens"):
        if not hasattr(GPUModelRunner, method):
            raise RuntimeError(f"Missing vLLM API {method}")


def validate_observations(array, rollouts, token_ids, vocab_size):
    if array.ndim != 2 or array.shape[1] != len(METRICS) + 3:
        raise ValueError("Observation schema mismatch")
    if len(array) != sum(len(ids) for ids in token_ids):
        raise ValueError("Observed positions do not equal returned generated tokens")
    rows, positions, ids = (array[:, i].astype(np.int64) for i in range(3))
    if np.any(rows < 0) or np.any(rows >= len(rollouts)):
        raise ValueError("Unknown request row")
    for row, expected in enumerate(token_ids):
        selected = np.flatnonzero(rows == row)
        order = np.argsort(positions[selected])
        selected = selected[order]
        if not np.array_equal(positions[selected], np.arange(1, len(expected) + 1)):
            raise ValueError(f"Missing, duplicated or prefill position in request {row}")
        if not np.array_equal(ids[selected], expected):
            raise ValueError(f"Sampler IDs differ from actual rollout IDs in request {row}")
    v = array[:, 3:]
    if not np.all(v[:, 0] + v[:, 1] == vocab_size):
        raise ValueError("Head and tail must cover full vocabulary")
    if not np.allclose(v[:, 2] + v[:, 3], 1, atol=3e-5, rtol=0):
        raise ValueError("Head and tail masses do not sum to one")
    if not np.isfinite(v[:, [0, 1, 2, 3, 4, 6, 7, 8, 13]]).all():
        raise ValueError("Invalid sampler probability/entropy")


def exact_batch_summary(values):
    out = {}
    for i, key in enumerate(METRICS):
        x = values[:, i]
        x = x[np.isfinite(x)]
        out[key] = {"n": len(x), "mean": float(x.mean(dtype=np.float64)) if len(x) else None,
                    "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))} if len(x) else {}}
    return out


def publish_batch(config, directory, stats, observation_path, rollouts, ids, next_cursor, identity):
    array = np.load(observation_path, allow_pickle=False)
    validate_observations(array, rollouts, ids, stats.vocab_size)
    values = array[:, 3:]
    stats.add(values, array[:, 1], array[:, 0], rollouts)
    batch = next_cursor["batches"]
    raw = directory / "raw"
    raw.mkdir(exist_ok=True)
    raw_path = raw / f"batch-{batch:09d}.npz"
    with raw_path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, observations=array,
                            rollouts=np.array(json.dumps(rollouts)),
                            columns=np.array(["request_row", "position_1based", "token_id", *METRICS]))
    os.replace(raw_path.with_suffix(".tmp"), raw_path)
    cursor = {**next_cursor, "fingerprint": identity["fingerprint"], "updated_at": time.time()}
    stats.save(directory / "state.npz", cursor)
    summary = {"cursor": cursor, **stats.summary()}
    atomic_json(directory / "summary.json", summary)
    batch_summary = {"batch": batch, "positions": len(array), "rollouts": rollouts,
                     "metrics_exact_this_batch": exact_batch_summary(values)}
    # Batch-specific files are idempotent across a crash/replay. A bounded window
    # of raw tokens permits later exact analysis; global histograms never expire.
    atomic_json(directory / "latest_batch.json", batch_summary)
    with (directory / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"batch": batch, "updated_at": cursor["updated_at"], "positions": len(array),
                            "head_mass": float(values[:, 2].mean(dtype=np.float64)),
                            "head_count": float(values[:, 0].mean(dtype=np.float64))}) + "\n")
    files = sorted(raw.glob("batch-*.npz"))
    total = sum(p.stat().st_size for p in files)
    limit = int(config["raw_gib_per_gpu"] * 1024**3)
    for path in files[:-1]:
        if total <= limit:
            break
        total -= path.stat().st_size
        path.unlink()
    Path(observation_path).unlink(missing_ok=True)
    log(f"Eval: GPU {config['gpus'][next_cursor['rank']]} committed batch={batch} "
        f"positions={len(array)} cumulative_positions={summary['positions']} "
        f"head_mass={summary['token_weighted']['head_mass']['mean']:.6f} "
        f"head_count={summary['token_weighted']['head_count']['mean']:.2f}")


def worker(config, rank):
    check_runtime()
    from vllm import LLM, SamplingParams
    identity, records = fingerprint(config)
    directory = Path(config["output"]) / f"gpu-{config['gpus'][rank]}"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.npz"
    stats, cursor = (StreamingStats.load(state_path) if state_path.exists() else
                     (StreamingStats(identity["vocab_size"]), {"epoch": 0, "offset": 0, "batches": 0,
                                                                 "rank": rank, "fingerprint": identity["fingerprint"]}))
    if cursor["fingerprint"] != identity["fingerprint"]:
        raise ValueError("Output belongs to a different experiment")
    stop = [False]
    def request_stop(*_):
        stop[0] = True
        log("Stop requested; finishing and checkpointing the current inference batch")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(f"Initializing persistent engine GPU {config['gpus'][rank]}; warmup/compile, ETA pending")
    engine = LLM(model=config["model"], dtype="bfloat16", tensor_parallel_size=1,
                 trust_remote_code=True, generation_config="vllm", seed=config["seed"] + rank,
                 max_model_len=config["max_prompt_tokens"] + config["max_tokens"],
                 gpu_memory_utilization=config["gpu_memory_utilization"],
                 max_num_seqs=config["max_num_seqs"], max_num_batched_tokens=config["max_num_batched_tokens"],
                 enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=False,
                 async_scheduling=False, disable_log_stats=False,
                 compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
                 worker_extension_cls="scripts.sampled_head_tail_worker.SampledHeadTailWorker")
    tokenizer = engine.get_tokenizer()
    installed = engine.collective_rpc("install_sampled_head_tail", args=(identity["vocab_size"], str(directory / "scratch")))
    log(f"Sampler installed: {installed}")
    prompt_cache = {}
    pending = None
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as publisher:
        while not stop[0]:
            if config.get("max_batches_per_gpu") and cursor["batches"] >= config["max_batches_per_gpu"]:
                break
            epoch, offset = cursor["epoch"], cursor["offset"]
            order = list(range(len(records)))
            random.Random(config["seed"] + epoch).shuffle(order)
            shard = order[rank::len(config["gpus"])]
            if offset >= len(shard):
                cursor.update(epoch=epoch + 1, offset=0)
                continue
            qids = shard[offset:offset + config["questions_per_batch"] // len(config["gpus"])]
            requests, params, meta = [], [], []
            for qid in qids:
                if qid not in prompt_cache:
                    prompt_cache[qid] = encode_prompt(tokenizer, records[qid]["question"], config["max_prompt_tokens"])
                prompt_ids, original_length = prompt_cache[qid]
                for replicate in range(config["rollouts_per_question"]):
                    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}/{epoch}/{qid}/{replicate}".encode()).digest()[:4], "little") % (2**31 - 1)
                    row = len(requests)
                    requests.append({"prompt_token_ids": prompt_ids})
                    params.append(SamplingParams(n=1, temperature=1., top_p=1., top_k=-1,
                                                 max_tokens=config["max_tokens"], seed=seed,
                                                 detokenize=False, ignore_eos=False,
                                                 extra_args={"head_tail_row": row}))
                    meta.append({"question_id": qid, "replicate": replicate, "seed": seed, "epoch": epoch,
                                 "prompt_tokens": len(prompt_ids), "prompt_original_tokens": original_length,
                                 "prompt_truncated": original_length > len(prompt_ids)})
            batch_id = cursor["batches"] + 1
            engine.collective_rpc("begin_sampled_head_tail", args=(batch_id,))
            batch_started = time.monotonic()
            outputs = engine.generate(requests, sampling_params=params, use_tqdm=True)
            observed = engine.collective_rpc("end_sampled_head_tail")[0]
            ids = [list(output.outputs[0].token_ids) for output in outputs]
            for row, output in enumerate(outputs):
                sample = output.outputs[0]
                meta[row].update(length=len(ids[row]), finish_reason=sample.finish_reason or "unknown",
                                 stop_reason=sample.stop_reason, terminal_token_id=ids[row][-1] if ids[row] else None)
            if pending is not None:
                pending.result()
            cursor = {"epoch": epoch, "offset": offset + len(qids), "batches": batch_id, "rank": rank,
                      "fingerprint": identity["fingerprint"]}
            pending = publisher.submit(publish_batch, config, directory, stats, observed["path"], meta, ids, cursor.copy(), identity)
            elapsed = time.monotonic() - batch_started
            epoch_batches = math.ceil(len(shard) / (config["questions_per_batch"] // len(config["gpus"])))
            completed = math.ceil(cursor["offset"] / (config["questions_per_batch"] // len(config["gpus"])))
            remaining = max(0, epoch_batches - completed) * elapsed
            log(f"Eval: GPU {config['gpus'][rank]} epoch={epoch+1} {completed}/{epoch_batches} batches "
                f"batch_seconds={elapsed:.2f} generated_tokens={sum(map(len,ids))} "
                f"tokens_per_second={sum(map(len,ids))/elapsed:.2f} epoch_remaining_seconds={remaining:.1f}; repeats continuously")
            log("GPU_PROGRESS " + json.dumps({"phase": "eval", "step": completed, "total": epoch_batches,
                                               "remaining_seconds": remaining, "updated_at": time.time(),
                                               "epoch": epoch+1, "continuous": not bool(config.get("max_batches_per_gpu"))}))
        if pending is not None:
            pending.result()
    log(f"Worker GPU {config['gpus'][rank]} checkpointed after {time.monotonic()-started:.1f}s")


def swanlab_run(config):
    if not config.get("swanlab", {}).get("enabled", False):
        return None
    try:
        import swanlab
        swanlab.init(project=config["swanlab"]["project"], workspace=config["swanlab"]["workspace"],
                     experiment_name=config["run_name"], config=config,
                     logdir=str(Path(config["output"]) / "swanlog"), mode="cloud")
        return swanlab
    except Exception as exc:
        log(f"SwanLab unavailable ({type(exc).__name__}); durable local statistics remain enabled")
        return None


def aggregate(config, identities, tracker, last_version):
    combined = StreamingStats(identities["vocab_size"])
    cursors = []
    for device in config["gpus"]:
        path = Path(config["output"]) / f"gpu-{device}" / "state.npz"
        if path.exists():
            part, cursor = StreamingStats.load(path)
            if cursor["fingerprint"] != identities["fingerprint"]:
                raise ValueError("Mismatched worker state")
            combined.merge(part)
            cursors.append(cursor)
    version = sum(c["batches"] for c in cursors)
    if version == last_version:
        return version
    summary = {"updated_at": time.time(), "worker_cursors": cursors, **combined.summary()}
    atomic_json(Path(config["output"]) / "summary.json", summary)
    combined.save(Path(config["output"]) / "global_state.npz", {"batches": version})
    compact = {"committed_batches": version, "positions": summary["positions"], "rollouts": summary["rollouts"],
               **{k: summary["token_weighted"][k]["mean"] for k in ("head_count", "tail_count", "head_mass", "tail_mass", "entropy_nats")}}
    with (Path(config["output"]) / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"updated_at": summary["updated_at"], **compact}) + "\n")
    log("GLOBAL " + json.dumps(compact))
    if tracker:
        metrics = {**compact, "response_length/mean": summary["response_length"]["mean"]}
        for key, values in summary["token_weighted"].items():
            metrics[f"token/{key}/mean"] = values["mean"]
            for q, value in values["quantiles"].items():
                metrics[f"token/{key}/p{100*float(q):g}"] = value.get("estimate") if isinstance(value, dict) else value
        for key, value in summary["fractions"].items():
            metrics["fraction/" + key] = value
        try:
            tracker.log({k: v for k, v in metrics.items() if v is not None}, step=version)
        except Exception as exc:
            log(f"SwanLab upload failed ({type(exc).__name__}); will continue local recording")
    return version


def supervisor(config_path, config):
    identity, _ = fingerprint(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "manifest.json"
    if existing.exists() and json.loads(existing.read_text())["fingerprint"] != identity["fingerprint"]:
        raise ValueError("Refusing to mix a different study into this output")
    atomic_json(existing, identity)
    tracker = None
    stop = [False]
    children = {}
    restarts = {rank: 0 for rank in range(len(config["gpus"]))}
    handles = []
    def request_stop(*_):
        stop[0] = True
        for process in children.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def start(rank):
        device = config["gpus"][rank]
        log_path = output / f"gpu-{device}.log"
        handle = log_path.open("a", buffering=1)
        handles.append(handle)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device), "GPU_LOG": str(log_path),
               "SPT_NOENV": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONUNBUFFERED": "1",
               "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
        children[rank] = subprocess.Popen([config["python"], "-u", "-m", "scripts.rollout_sampled_head_tail",
                                          "--config", str(config_path), "--worker", str(rank)],
                                         env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"Started GPU {device} worker pid={children[rank].pid}, log={log_path}")
    try:
        for rank in restarts:
            start(rank)
        # Network logging must never delay GPU worker initialization.
        tracker = swanlab_run(config)
        version, last = -1, 0.
        while not stop[0]:
            all_done = True
            for rank, process in list(children.items()):
                status = process.poll()
                if status is None:
                    all_done = False
                elif status != 0:
                    if restarts[rank] >= 2:
                        raise RuntimeError(f"GPU worker {rank} failed three times; see its log")
                    restarts[rank] += 1
                    log(f"Worker {rank} failed exit={status}; resuming committed cursor, restart {restarts[rank]}")
                    start(rank)
                    all_done = False
            if time.monotonic() - last > config["summary_interval_seconds"] or all_done:
                version = aggregate(config, identity, tracker, version)
                last = time.monotonic()
            if all_done:
                break
            time.sleep(2)
    finally:
        request_stop()
        for process in children.values():
            try:
                process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
        aggregate(config, identity, tracker, -1)
        for handle in handles:
            handle.close()
        if tracker:
            tracker.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.preflight:
        check_runtime()
        identity, _ = fingerprint(config)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "1+1?"}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        if "<think>\n\n</think>" not in prompt:
            raise ValueError("No-thinking template was not applied")
        encode_prompt(tokenizer, "1+1?", config["max_prompt_tokens"])
        print(json.dumps({"preflight": "passed", "questions": identity["questions"],
                          "vocab_size": identity["vocab_size"], "fingerprint": identity["fingerprint"]}))
    elif args.worker is not None:
        worker(config, args.worker)
    else:
        required = {"GPU_OWNER": "heqiangqiang", "GPU_TASK": "logic RL", "GPU_KIND": "eval"}
        if any(os.environ.get(k) != v for k, v in required.items()) or not os.environ.get("GPU_LOG") or not os.environ.get("GPU_HOURS"):
            raise ValueError("Missing or incorrect shared-GPU registration")
        supervisor(args.config.resolve(), config)


if __name__ == "__main__":
    main()
