#!/usr/bin/env bash
# Queue-ready baseline job; GPU_DEVICES is supplied by the team scheduler.
set -euo pipefail
[[ "$(hostname)" == VM-8-245-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-eval8b-step50-avg32-20261107
python_bin=/data/heqiangqiang/miniforge3/envs/rlvr/bin/python
cd "$project"
export GPU_OWNER=heqiangqiang GPU_TASK=Logic_RL GPU_KIND=eval
export GPU_HOURS="${GPU_HOURS:-4}" GPU_RUN=klopsd8b-base-avg32-5bench-20261108
export GPU_LOG="$project/outputs/.logs/$GPU_RUN.log"
export FRUGALOPL_PYTHON="$python_bin" FRUGALOPL_TMUX_SESSION="$GPU_RUN"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
runtime_config="$project/outputs/kl_opsd_base_avg32_runtime_config.json"
CUDA_VISIBLE_DEVICES="" "$python_bin" - <<'PY'
import csv, json, os, subprocess
from pathlib import Path
from utils.eval import load_evaluation_config
selected = [int(value.strip()) for value in os.environ['GPU_DEVICES'].split(',')]
assert len(selected) == len(set(selected)) == 4 and min(selected) >= 0
config = load_evaluation_config('configs/eval/kl_opsd_qwen3_8b_base_avg32_5bench_4GPU.yaml')
assert config['actor']['lora']['enabled'] is False and config['actor']['lora']['path'] is None
assert config['actor']['evaluation']['rollouts_per_prompt'] == 32
gpus = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()))
apps = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).splitlines()))
occupied = {r[0].strip() for r in apps if r}
available = {int(i): (u.strip(), int(m)) for i, u, m in gpus}
for index in selected:
    uuid, used = available[index]
    assert uuid not in occupied and used < 1024, f'GPU {index} is occupied; refusing to launch'
output = Path(config['run']['output_dir'])
assert not (output / 'summary.json').exists(), 'Baseline already completed'
assert not any((output / 'rollouts').glob('batch_*.json')), 'An earlier attempt needs review'
config['actor']['gpu_ids'] = selected
Path('outputs').mkdir(exist_ok=True)
Path('outputs/kl_opsd_base_avg32_runtime_config.json').write_text(json.dumps(config, indent=2))
print('Verified base model, matching Avg@32 settings, empty GPUs:', selected, flush=True)
PY
cache=/data/heqiangqiang/.cache/frugalopl-eval
export TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export TORCH_EXTENSIONS_DIR="$cache/torch_extensions" VLLM_CACHE_ROOT="$cache/vllm"
export HF_HOME="$cache/huggingface" CUDA_CACHE_PATH="$cache/cuda" TORCH_HOME="$cache/torch"
unset CUDA_VISIBLE_DEVICES DS_ACCELERATOR
bash scripts/start_eval.sh "$runtime_config"
