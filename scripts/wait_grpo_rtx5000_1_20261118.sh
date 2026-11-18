#!/usr/bin/env bash
set -euo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec 8> outputs/.logs/grpo-watch-20261118.lock
flock -n 8 || exit 0
echo "$(date -u +%FT%TZ) Watching rtx5000-1 for four free GPUs, every 2 seconds."
while true; do
    if selected="$(python3 - <<'PY'
import csv,subprocess
def query(fields,category):
    return list(csv.reader(subprocess.check_output(["nvidia-smi",f"--query-{category}={fields}","--format=csv,noheader,nounits"],text=True).splitlines(),skipinitialspace=True))
busy={row[0].strip() for row in query("gpu_uuid","compute-apps")}
free=sorted(int(row[0]) for row in query("index,uuid,memory.used","gpu") if row[1].strip() not in busy and int(row[2])<1024)
if len(free)<4: raise SystemExit(1)
print(",".join(map(str,free[:4])))
PY
)"; then
        GPU_DEVICES="$selected" bash scripts/launch_grpo_rtx5000_1_20261118.sh
        exit $?
    fi
    sleep 2
done
