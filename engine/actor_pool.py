"""Claim complete, unstarted training shards on any available Actor."""
from __future__ import annotations

import copy
import json
import pickle
import time

from utils.eval import write_json


def pool_keys(root, round_id):
    prefix = f"{root}:actor_pool:{round_id}"
    return {name: f"{prefix}:{name}" for name in ("ready", "claims", "completed")}


def completed_tasks(client, keys):
    return [json.loads(raw) for raw in client.hgetall(keys["completed"]).values()]


def run_actor_pool(actor, client, envelope, stop_event):
    if actor.round_gate is None or envelope.get("kind") != "actor_pool":
        raise ValueError("Actor task pools require a shared GPU round.")
    round_id = envelope["round_id"]
    keys = pool_keys(actor.queue_keys()["actor_train"].removesuffix(":actor:train"), round_id)
    count = envelope.get("task_count")
    if (envelope.get("pool_keys") != keys or isinstance(count, bool)
            or not isinstance(count, int) or count < 1):
        raise ValueError("Actor task pool does not match its run/round manifest.")
    if not actor.round_gate.wait_actor(actor.gpu_id, round_id, stop_event):
        return
    success = False
    try:
        while not stop_event.is_set():
            raw = client.rpop(keys["ready"])
            if raw is None:
                success = True
                break
            task = pickle.loads(raw)
            if any(task.get(key) != envelope.get(key) for key in ("wave_id", "round_id", "policy_version", "batch_ids")):
                raise RuntimeError("Actor pool task has a stale wave or policy version.")
            if len(task.get("batches", [])) != 1:
                raise RuntimeError("Actor pool tasks must contain exactly one complete question shard.")
            index = task["batches"][0]["actor_shard"]["index"]
            if (task["batches"][0]["actor_shard"].get("count") != count
                    or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count
                    or not client.hsetnx(keys["claims"], index, actor.gpu_id)):
                raise RuntimeError("Duplicate or invalid Actor pool task claim.")
            task = copy.deepcopy(task)
            task["batches"][0]["actor_shard"]["assigned_gpu_id"] = actor.gpu_id
            started = time.monotonic()
            actor._process_training_wave(client, task, stop_event=stop_event, pooled=True)
            if stop_event.is_set():
                return
            client.hset(keys["completed"], index, json.dumps({
                "shard_index": index, "gpu_id": actor.gpu_id, "started_monotonic": started,
                "finished_monotonic": time.monotonic(), "policy_version": envelope["policy_version"]}))
    finally:
        # Retain the live engine between claimed shards. No Learner is
        # admitted on this card until this final sleep and acknowledgement.
        if actor.gpu_gate.acquire(stop_event):
            try:
                if not actor._sleep_engine_after_turn():
                    raise RuntimeError("Actor pool could not sleep before its GPU handoff.")
            finally:
                actor.gpu_gate.release()
        elif success:
            success = False
    if success:
        actor.round_gate.actor_done(actor.gpu_id, round_id)
        print(f"[Actor GPU {actor.gpu_id}] Drained shared rollout pool for round={round_id}.", flush=True)


def archive_completed_pool(controller, client, dispatch):
    info = dispatch.get("actor_pool")
    if info is None:
        return
    keys, count = info["keys"], info["task_count"]
    claims = {int(key): int(value) for key, value in client.hgetall(keys["claims"]).items()}
    completed = completed_tasks(client, keys)
    if (set(claims) != set(range(count)) or {row["shard_index"] for row in completed} != set(claims)
            or len(completed) != count or client.llen(keys["ready"])):
        raise RuntimeError("Actor task pool did not process its exact global batch once.")
    if any(row["gpu_id"] != claims[row["shard_index"]] or row["policy_version"] != info["policy_version"]
           for row in completed):
        raise RuntimeError("Actor task pool completion disagrees with its task claims.")
    write_json(controller.output_dir / "actor_pool" / f"round_{dispatch['round_id']:06d}.json",
               {"task_count": count, "policy_version": info["policy_version"], "claims": claims,
                "completed": sorted(completed, key=lambda row: row["shard_index"])})
    client.delete(*keys.values())


def cleanup_actor_pools(client, root):
    """Discard only this run's abandoned work, after all Workers stop."""
    prefix = f"{root}:actor_pool:"
    try:
        keys = [key for key in client.scan_iter(match=f"{prefix}*")
                if (key.decode() if isinstance(key, bytes) else str(key)).startswith(prefix)]
        if keys:
            client.delete(*keys)
    except Exception as error:
        # A failed Redis connection must not mask the original run failure
        # or prevent the Controller from closing its SwanLab experiment.
        print(f"[Controller] Actor pool key cleanup failed: {type(error).__name__}.", flush=True)
