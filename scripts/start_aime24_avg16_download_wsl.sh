#!/usr/bin/env bash
set -euo pipefail
project=/home/qqh/projects/FrugalOPL
python_bin=/home/qqh/miniconda3/envs/rlvr/bin/python
output=/mnt/c/Users/qqian/Desktop/Projects/FrugalOPL/outputs/aime24_avg16_rtx5000_1_20261108
session=aime24-avg16-download-20261109
mkdir -p "$output"
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Download watcher already running: $session"
    exit 0
fi
printf -v command '%q ' env CUDA_VISIBLE_DEVICES= "$python_bin" -u \
    "$project/scripts/aime24_avg16_transfer.py" watch --output "$output" --poll 120
printf -v log '%q' "$output/transfer.log"
printf -v status '%q' "$output/transfer.exit"
inner="${command}>${log} 2>&1; result=\$?; printf '%s\\n' \"\$result\" >${status}; exit \"\$result\""
printf -v quoted '%q' "$inner"
tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted"
echo "Started local CPU-only download watcher: $session"
echo "Destination: $output"
