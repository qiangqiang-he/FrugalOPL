#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == VM-8-172-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-aime24-avg16-20261108
session=aime24-half-hour-monitor-20261109
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Half-hour supervisor already running: $session"
    exit 0
fi
printf -v command '%q ' env CUDA_VISIBLE_DEVICES= /data/heqiangqiang/miniforge3/envs/rlvr/bin/python -u \
    "$project/tests/helpers/aime24_avg16_monitor.py"
printf -v log '%q' "$project/outputs/.logs/half_hour_monitor.log"
inner="exec ${command}>>${log} 2>&1"
printf -v quoted '%q' "$inner"
tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted"
echo "Started half-hour supervisor: $session"
