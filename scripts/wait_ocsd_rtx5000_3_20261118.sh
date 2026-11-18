#!/usr/bin/env bash
set -euo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(hostname)" == VM-8-245-ubuntu ]] || exit 1

config=configs/training/ocsd/rtx5000_3_20261118_4GPU.yaml
log_dir="$PWD/outputs/.logs"
mkdir -p "$log_dir"
exec 9>"$log_dir/ocsd-rtx5000-3-20261118.lock"
flock -n 9 || exit 0

set +u
source /data/heqiangqiang/miniforge3/etc/profile.d/conda.sh
conda activate rlvr
set -u
export FRUGALOPL_PYTHON="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX"
export GPU_OWNER=heqiangqiang GPU_TASK="logic RL" GPU_HOURS=24 GPU_KIND=train
export GPU_RUN=ocsd-gamma025-rtx5000-3-20261118
export GPU_LOG="$log_dir/ocsd-rtx5000-3-20261118.log"
export FRUGALOPL_TMUX_SESSION=ocsd-rtx5000-3-20261118
export FRUGALOPL_SWANLAB_ENV_FILE=/home/ubuntu/heqiangqiang/.config/frugalopl/swanlab.env
[[ -r "$config" && -r "$FRUGALOPL_SWANLAB_ENV_FILE" ]]
if tmux has-session -t "=$FRUGALOPL_TMUX_SESSION" 2>/dev/null; then exit 0; fi

cards=0,1,2,3
printf '%s Waiting for RTX5000-3 GPUs %s every 2 seconds\n' "$(date -u +%FT%TZ)" "$cards"
while true; do
    if devices="$(nvidia-smi -i "$cards" --query-gpu=uuid,memory.used --format=csv,noheader,nounits)" &&
        processes="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits)" &&
        awk -F ', *' 'NR == FNR {busy[$0] = 1; next}
            $1 !~ /^GPU-/ || $2 !~ /^[0-9]+$/ || $2 >= 1024 || ($1 in busy) {exit 1}
            {free++}
            END {if (free != 4) exit 1}' <(printf '%s\n' "$processes") <(printf '%s\n' "$devices"); then
        printf '%s GPUs %s are free. Launching OCSD.\n' "$(date -u +%FT%TZ)" "$cards"
        exec bash scripts/start_train.sh "$config"
    fi
    sleep 2
done
