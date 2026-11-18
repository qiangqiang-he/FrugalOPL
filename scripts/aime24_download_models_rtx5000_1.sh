#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == VM-8-172-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-aime24-avg16-20261108
cd "$project"
sha256sum tests/helpers/download_model_snapshot.py
for size in 1.7B 8B; do
    if [[ -f "/data/heqiangqiang/cache/models/Qwen3-$size/download_manifest.json" ]]; then
        continue
    fi
    session="aime24-download-${size//./p}-20261108"
    if tmux has-session -t "=$session" 2>/dev/null; then
        continue
    fi
    printf -v command '%q ' env CUDA_VISIBLE_DEVICES= /data/heqiangqiang/miniforge3/envs/rlvr/bin/python -u \
        tests/helpers/download_model_snapshot.py "Qwen/Qwen3-$size" "/data/heqiangqiang/cache/models/Qwen3-$size"
    printf -v log '%q' "$project/outputs/.logs/download-$size.log"
    printf -v status '%q' "$project/outputs/.logs/download-$size.status"
    rm -f -- "$project/outputs/.logs/download-$size.status"
    inner="${command}>${log} 2>&1; result=\$?; printf '%s\\n' \"\$result\" >${status}; exit \"\$result\""
    printf -v quoted '%q' "$inner"
    tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted"
done
tmux list-sessions -F '#S'
