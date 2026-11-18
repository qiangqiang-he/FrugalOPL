"""Prepared one-shot handoff of our explicit GRPO run to the inference study.

Run only after deployment/preflight. Never signals a process from another run.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml


def environment(pid):
    try:
        return dict(part.split("=", 1) for part in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in part)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--controller", required=True, type=int)
    parser.add_argument("--old-run", required=True)
    parser.add_argument("--new-run", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    python = config["python"]
    # Last CPU preflight happens before any stop signal.
    subprocess.run([python, "-m", "scripts.rollout_sampled_head_tail", "--config", str(args.config), "--preflight"], check=True)
    old = environment(args.controller)
    if old.get("GPU_OWNER") != "heqiangqiang" or old.get("GPU_RUN") != args.old_run:
        raise SystemExit("Controller ownership/run changed; handoff cancelled")
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    devices = {int(i): u.strip() for i, u, _ in (line.split(",") for line in gpu_text.splitlines())}
    selected = {devices[g] for g in config["gpus"]}
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
    for line in compute.splitlines():
        pid, uuid = (v.strip() for v in line.split(","))
        if uuid in selected:
            env = environment(int(pid))
            if env.get("GPU_OWNER") != "heqiangqiang" or env.get("GPU_RUN") != args.old_run:
                raise SystemExit(f"GPU process {pid} is not from the authorized old run")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    new = {**os.environ, "GPU_OWNER": "heqiangqiang", "GPU_TASK": "logic RL", "GPU_KIND": "eval",
           "GPU_HOURS": "24", "GPU_RUN": args.new_run, "GPU_LOG": str(output / "supervisor.log"),
           "FRUGALOPL_PYTHON": python,
           "FRUGALOPL_SWANLAB_ENV_FILE": "/home/ubuntu/heqiangqiang/.config/frugalopl/swanlab.env"}
    print("Preflight passed; stopping only the verified old run. Existing durable adapter/optimizer checkpoints are preserved.", flush=True)
    os.kill(args.controller, signal.SIGINT)
    began = time.monotonic()
    while True:
        gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
        compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        occupied = {line.split(",")[1].strip() for line in compute.splitlines() if "," in line}
        info = {int(i): (u.strip(), int(m)) for i,u,m in (line.split(",") for line in gpu_text.splitlines())}
        if all(info[g][0] not in occupied and info[g][1] < 1024 for g in config["gpus"]):
            break
        if time.monotonic() - began > 120:
            # Escalation is limited to the same explicitly identified old run.
            for proc in Path("/proc").iterdir():
                if proc.name.isdigit():
                    env = environment(int(proc.name))
                    if env.get("GPU_OWNER") == "heqiangqiang" and env.get("GPU_RUN") == args.old_run:
                        try:
                            os.kill(int(proc.name), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
        if time.monotonic() - began > 180:
            raise SystemExit("Old run has not released all four GPUs; no other process was touched")
        time.sleep(1)
    print(f"All selected GPUs released after {time.monotonic()-began:.1f}s; immediately launching continuous inference.", flush=True)
    with (output / "supervisor.log").open("ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.execvpe("bash", ["bash", "scripts/start_sampled_head_tail.sh", str(args.config)], new)


if __name__ == "__main__":
    main()
