#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
[[ "$(hostname)" == VM-8-172-ubuntu ]] || exit 1
mkdir -p outputs/.logs
exec 9> outputs/.logs/grpo-rtx5000-1-20261118.lock
flock -n 9 || exit 0
set +u
source /data/heqiangqiang/miniforge3/etc/profile.d/conda.sh
conda activate frugalopl
set -u
export FRUGALOPL_PYTHON="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX"
export GPU_OWNER=heqiangqiang GPU_TASK="logic RL" GPU_KIND=train
export GPU_HOURS="${GPU_HOURS:-24}"
export GPU_RUN="${GPU_RUN:-grpo-rtx5000-1-$(date -u +%Y%m%dT%H%M%SZ)}"
export GPU_LOG="$project_root/outputs/.logs/grpo-rtx5000-1-20261118.log"
export FRUGALOPL_TMUX_SESSION=grpo-rtx5000-1-20261118
export FRUGALOPL_SWANLAB_ENV_FILE=/home/ubuntu/heqiangqiang/.config/frugalopl/tail-sup-swanlab.env
export TRITON_CACHE_DIR=/data/heqiangqiang/cache/rlvr/triton
export TORCH_EXTENSIONS_DIR=/data/heqiangqiang/cache/rlvr/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/data/heqiangqiang/cache/rlvr/torchinductor
export CUDA_CACHE_PATH=/data/heqiangqiang/cache/rlvr/cuda
export VLLM_CACHE_ROOT=/data/heqiangqiang/cache/rlvr/vllm
if tmux has-session -t "=$FRUGALOPL_TMUX_SESSION" 2>/dev/null; then exit 0; fi
selected="$(GPU_DEVICES="${GPU_DEVICES:-}" "$FRUGALOPL_PYTHON" - <<'PY'
import csv,os,subprocess
def query(fields,category):
    return list(csv.reader(subprocess.check_output(["nvidia-smi",f"--query-{category}={fields}","--format=csv,noheader,nounits"],text=True).splitlines(),skipinitialspace=True))
busy={row[0].strip() for row in query("gpu_uuid","compute-apps")}
free=sorted(int(row[0]) for row in query("index,uuid,memory.used","gpu") if row[1].strip() not in busy and int(row[2])<1024)
requested=os.environ.get("GPU_DEVICES","")
ids=sorted(map(int,requested.split(","))) if requested else free[:4]
if len(ids)!=4 or len(set(ids))!=4 or not set(ids)<=set(free):
    raise SystemExit("Four assigned GPUs are not free; refusing to launch.")
print("_".join(map(str,ids)))
PY
)"
config="configs/training/grpo/rtx5000_1_20261118/gpu_${selected}.yaml"
[[ -r "$config" && -r "$FRUGALOPL_SWANLAB_ENV_FILE" ]] || exit 1
redis_dir="$project_root/outputs/.redis"
mkdir -p "$redis_dir"
if ! "$CONDA_PREFIX/bin/redis-cli" -p 16429 ping >/dev/null 2>&1; then
    "$CONDA_PREFIX/bin/redis-server" --bind 127.0.0.1 --port 16429 --protected-mode yes --save '' --appendonly no --daemonize yes --dir "$redis_dir" --pidfile "$redis_dir/redis.pid" --logfile "$redis_dir/redis.log"
fi
printf '%s Starting GRPO on GPUs %s\n' "$(date -u +%FT%TZ)" "$selected"
bash scripts/start_train.sh "$config"
