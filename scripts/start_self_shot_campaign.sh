#!/usr/bin/env bash
# Start/re-attach the CPU-only serial supervisor. Training has its own tmux.
set -euo pipefail
[[ "$(hostname)" == VM-8-245-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-self-shot-opsd-20261107
session=self-shot-opsd-six-runs-20261107
python_bin=/data/heqiangqiang/miniforge3/envs/rlvr/bin/python
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Campaign already running: $session"
    exit 0
fi
mkdir -p "$project/outputs/.logs"
printf -v python_command '%q ' "$python_bin" scripts/self_shot_campaign.py --project "$project" --gpu-devices "${GPU_DEVICES:-0,1,2,3}"
printf -v quoted_log '%q' "$project/outputs/.logs/campaign.log"
script="set -o pipefail; source /data/heqiangqiang/miniforge3/etc/profile.d/conda.sh; conda activate rlvr; unset CUDA_VISIBLE_DEVICES DS_ACCELERATOR; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; $python_command 2>&1 | tee -a $quoted_log; exit \${PIPESTATUS[0]}"
printf -v quoted_script '%q' "$script"
tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted_script"
echo "Campaign supervisor started: $session"
