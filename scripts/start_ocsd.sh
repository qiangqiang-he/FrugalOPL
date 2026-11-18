#!/usr/bin/env bash
set -euo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

: "${GPU_DEVICES:?Set GPU_DEVICES to four physical GPU IDs, e.g. 0,1,2,3}"
: "${GAMMA_POS:?Set GAMMA_POS to the positive transfer coefficient, e.g. 0.2}"
python_bin="${FRUGALOPL_PYTHON:-python}"
if ! FRUGALOPL_PYTHON="$(command -v "$python_bin")"; then
    echo "Activate the FrugalOPL Python environment or set FRUGALOPL_PYTHON." >&2
    exit 1
fi
export FRUGALOPL_PYTHON

launch="$("$FRUGALOPL_PYTHON" - "$GPU_DEVICES" "$GAMMA_POS" <<'PY'
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from engine.actor import load_merged_config

raw_devices, raw_gamma = sys.argv[1:]
if re.fullmatch(r"[0-9]+(?:,[0-9]+){3}", raw_devices) is None:
    raise SystemExit("GPU_DEVICES must be four comma-separated physical GPU IDs.")
gpu_ids = [int(value) for value in raw_devices.split(",")]
if len(set(gpu_ids)) != 4:
    raise SystemExit("GPU_DEVICES must contain four distinct IDs.")
try:
    gamma = float(raw_gamma)
except ValueError:
    raise SystemExit("GAMMA_POS must be a number in [0, 1].") from None
if not 0.0 <= gamma <= 1.0:
    raise SystemExit("GAMMA_POS must be a number in [0, 1].")

profile = Path("configs/training/ocsd/qwen3_1p7b_tau098_4GPU.yaml").resolve()
config = load_merged_config(profile)
if config["actor"]["engine"]["enforce_eager"] is not False:
    raise SystemExit("OCSD requires enforce_eager: false for CUDA Graph.")

devices = ",".join(str(gpu_id) for gpu_id in gpu_ids)
status = subprocess.check_output(["nvidia-smi", "-q", "-x", "-i", devices], text=True)
gpus = ET.fromstring(status).findall("gpu")
if len(gpus) != 4:
    raise SystemExit("Could not verify the four selected GPUs.")
for gpu in gpus:
    memory = re.fullmatch(r"\s*(\d+)\s+MiB\s*", gpu.findtext("fb_memory_usage/used", ""))
    processes = gpu.find("processes")
    if memory is None or processes is None or (processes.text or "").strip() or len(processes):
        raise SystemExit("A selected GPU has processes or its availability could not be verified.")
    if int(memory[1]) >= 1024:
        raise SystemExit("A selected GPU already uses at least 1 GiB of memory.")

gamma_tag = str(gamma).replace(".", "p")
run_name = f"{config['run']['name']}_gamma{gamma_tag}"
# Disjoint sets on the same host get distinct rendezvous ports.
master_port = config["learner"]["distributed"]["master_port"] + min(gpu_ids)
run = {"name": run_name}
if model_path := os.environ.get("OCSD_MODEL_PATH"):
    run["policy"] = {"model_name_or_path": model_path}
if redis_url := os.environ.get("OCSD_REDIS_URL"):
    run["redis"] = {"url": redis_url}
overlay = {
    "extends": str(profile),
    "run": run,
    "actor": {"gpu_ids": gpu_ids},
    "learner": {"gpu_ids": gpu_ids, "distributed": {"master_port": master_port}},
    "algorithm": {"ocsd_gamma": gamma},
}
directory = Path("outputs/.launch")
directory.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=directory, prefix=f"{run_name}-", suffix=".yaml", delete=False,
) as file:
    yaml.safe_dump(overlay, file, sort_keys=False)
    print(run_name)
    print(file.name)
PY
)"

export FRUGALOPL_TMUX_SESSION="${launch%%$'\n'*}"
launch_config="${launch#*$'\n'}"
export GPU_OWNER=heqiangqiang GPU_TASK="logic RL" GPU_KIND=train GPU_HOURS="${GPU_HOURS:-24}"
export GPU_RUN="${FRUGALOPL_TMUX_SESSION}-$(hostname -s)"
exec bash scripts/start_train.sh "$launch_config"
