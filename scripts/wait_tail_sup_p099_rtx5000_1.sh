#!/usr/bin/env bash
# CPU-only watcher. Launch real registered training as soon as GPUs 4-7 are free.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
[[ "$(hostname)" == VM-8-172-ubuntu ]] || exit 1
echo "$(date -u +%FT%TZ) Waiting for GPUs 4,5,6,7; checking every 2 seconds."
while true; do
    if python3 - <<'PY'
import csv
import subprocess

def query(fields, category):
    text = subprocess.check_output(["nvidia-smi", f"--query-{category}={fields}",
                                    "--format=csv,noheader,nounits"], text=True)
    return list(csv.reader(text.splitlines(), skipinitialspace=True))

busy = {row[0].strip() for row in query("gpu_uuid", "compute-apps")}
gpus = {int(row[0]): (row[1].strip(), int(row[2])) for row in query("index,uuid,memory.used", "gpu")}
raise SystemExit(0 if all(gpu in gpus and gpus[gpu][0] not in busy and gpus[gpu][1] < 1024
                         for gpu in range(4, 8)) else 1)
PY
    then
        echo "$(date -u +%FT%TZ) GPUs 4-7 are free; starting top-p=0.99."
        exec bash scripts/start_tail_sup_p099_rtx5000_1.sh
    fi
    sleep 2
done
