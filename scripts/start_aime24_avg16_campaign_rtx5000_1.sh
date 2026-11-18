#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == VM-8-172-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-aime24-avg16-20261108
cd "$project"
sha256sum -c --quiet aime24-source-manifest.sha256
[[ -x /data/heqiangqiang/miniforge3/envs/rlvr/bin/python ]]
[[ -x /data/heqiangqiang/miniforge3/envs/rlvr/bin/nvcc ]]
session=aime24-avg16-campaign-20261108
if tmux has-session -t "=$session" 2>/dev/null; then
    echo 'Campaign already running'
    exit 0
fi
printf -v command '%q ' env CUDA_VISIBLE_DEVICES= /data/heqiangqiang/miniforge3/envs/rlvr/bin/python -u \
    scripts/aime24_avg16_campaign_rtx5000_1.py
printf -v log '%q' "$project/outputs/.logs/campaign.log"
printf -v status '%q' "$project/outputs/.logs/campaign.status"
inner="${command}>${log} 2>&1; result=\$?; printf '%s\\n' \"\$result\" >${status}; exit \"\$result\""
printf -v quoted '%q' "$inner"
tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted"
echo "Started $session; log: $project/outputs/.logs/campaign.log"
