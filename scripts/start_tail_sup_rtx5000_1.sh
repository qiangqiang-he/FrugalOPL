#!/usr/bin/env bash
# Launch only the approved p=0.9 experiment; p=0.95 remains a separate config.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
[[ "$(hostname)" == VM-8-172-ubuntu ]] || { echo 'Expected rtx5000-1.' >&2; exit 1; }

set +u
source /data/heqiangqiang/miniforge3/etc/profile.d/conda.sh
conda activate frugalopl
set -u
export FRUGALOPL_PYTHON="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX"
export GPU_OWNER=heqiangqiang GPU_TASK="logic RL" GPU_KIND=train
# Initial rough estimate for 100 updates of up to 256 * 4096 response tokens.
# Preparation, early EOS and full-vocabulary scoring make this uncertain;
# replace the estimate using measured training progress at first inspection.
export GPU_HOURS="${GPU_HOURS:-24}"
export GPU_RUN="${GPU_RUN:-tail-sup-p090-$(date -u +%Y%m%dT%H%M%SZ)}"
export GPU_LOG="$project_root/outputs/.logs/tail-sup-p090-20261117.log"
export FRUGALOPL_TMUX_SESSION=tail-sup-p090-20261117
export FRUGALOPL_SWANLAB_ENV_FILE=/home/ubuntu/heqiangqiang/.config/frugalopl/tail-sup-swanlab.env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/data/heqiangqiang/cache/rlvr/triton
export TORCH_EXTENSIONS_DIR=/data/heqiangqiang/cache/rlvr/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/data/heqiangqiang/cache/rlvr/torchinductor
export CUDA_CACHE_PATH=/data/heqiangqiang/cache/rlvr/cuda
export VLLM_CACHE_ROOT=/data/heqiangqiang/cache/rlvr/vllm
mkdir -p -- "$project_root/outputs/.logs"

if tmux has-session -t "=$FRUGALOPL_TMUX_SESSION" 2>/dev/null; then
    echo 'The tail_sup p=0.9 session already exists.' >&2
    exit 1
fi
"$FRUGALOPL_PYTHON" - <<'PY'
import csv
import subprocess

def query(fields, category):
    result = subprocess.check_output(["nvidia-smi", f"--query-{category}={fields}",
                                      "--format=csv,noheader,nounits"], text=True)
    return list(csv.reader(result.splitlines(), skipinitialspace=True))

busy = {row[0].strip() for row in query("gpu_uuid", "compute-apps")}
gpus = {int(row[0]): (row[1].strip(), int(row[2])) for row in query("index,uuid,memory.used", "gpu")}
for gpu in range(4):
    uuid, memory = gpus[gpu]
    if uuid in busy or memory >= 1024:
        raise SystemExit(f"GPU {gpu} is occupied; refusing to launch.")
print("Confirmed GPUs 0,1,2,3 have no compute processes and memory < 1 GiB.")
PY

redis_dir="$project_root/outputs/.redis"
mkdir -p -- "$redis_dir"
if "$CONDA_PREFIX/bin/redis-cli" -h 127.0.0.1 -p 16419 ping >/dev/null 2>&1; then
    [[ -f "$redis_dir/redis.pid" ]] || { echo 'Redis port 16419 belongs to an existing service.' >&2; exit 1; }
else
    "$CONDA_PREFIX/bin/redis-server" --bind 127.0.0.1 --port 16419 --protected-mode yes \
        --save '' --appendonly no --daemonize yes --dir "$redis_dir" \
        --pidfile "$redis_dir/redis.pid" --logfile "$redis_dir/redis.log"
fi
bash scripts/start_train.sh configs/training/tail_sup/qwen3_1p7b_rtx5000_1_top_p_090_4GPU.yaml
