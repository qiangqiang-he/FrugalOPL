#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPU_OWNER:?}" "${GPU_TASK:?}" "${GPU_HOURS:?}" "${GPU_KIND:?}" "${GPU_LOG:?}" "${GPU_RUN:?}"
[[ "$GPU_OWNER" == heqiangqiang && "$GPU_TASK" == 'logic RL' && "$GPU_KIND" == eval ]]
export GPU_OWNER GPU_TASK GPU_HOURS GPU_KIND GPU_LOG GPU_RUN
export SPT_NOENV=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Shared ubuntu's default Triton cache may contain another user's files.
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=4
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
config="${1:?supply a YAML config}"
python_bin="${FRUGALOPL_PYTHON:-/data/heqiangqiang/miniforge3/envs/rlvr/bin/python}"
if [[ -n "${FRUGALOPL_SWANLAB_ENV_FILE:-}" && -f "$FRUGALOPL_SWANLAB_ENV_FILE" ]]; then
  set -a
  source "$FRUGALOPL_SWANLAB_ENV_FILE"
  set +a
fi
"$python_bin" - "$config" <<'PY'
import subprocess, sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()
info = {int(i.strip()): (u.strip(), int(m.strip())) for i,u,m in (line.split(',') for line in gpu_rows.splitlines())}
for device in config['gpus']:
    uuid, memory = info[device]
    if uuid in processes or memory >= 1024:
        raise SystemExit(f'GPU {device} is occupied; refusing to interfere')
print('Selected GPUs are free:', config['gpus'], flush=True)
PY
exec "$python_bin" -u -m scripts.rollout_sampled_head_tail --config "$config"
