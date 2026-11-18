#!/usr/bin/env bash
# Deploy only this evaluation to the explicitly selected RTX5000-3 host.
set -euo pipefail
[[ "$(hostname)" == VM-8-245-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-eval8b-step50-avg32-20261107
archive=/home/ubuntu/heqiangqiang/frugalopl-eval8b-step50-20261107.tar.gz
python_bin=/data/heqiangqiang/miniforge3/envs/rlvr/bin/python
config=configs/eval/kl_opsd_qwen3_8b_step50_avg32_5bench_4GPU.yaml
mkdir -p "$project"
tar -xzf "$archive" -C "$project"
cd "$project"
bash -n scripts/start_eval.sh
CUDA_VISIBLE_DEVICES="" DS_ACCELERATOR=cpu "$python_bin" - <<'PY'
from pathlib import Path
import json, yaml
from utils.eval import load_evaluation_config, load_dataset_records
path = Path('configs/eval/kl_opsd_qwen3_8b_step50_avg32_5bench_4GPU.yaml')
c = load_evaluation_config(path)
assert Path(c['run']['policy']['model_name_or_path'], 'config.json').is_file()
adapter = Path(c['actor']['lora']['path'])
assert (adapter / 'adapter_model.safetensors').stat().st_size == 349244256
assert json.loads((adapter / 'adapter_config.json').read_text())['r'] == 64
counts = {name: len(load_dataset_records(v, config_path=path))
          for name, v in c['actor']['evaluation']['datasets'].items()}
assert counts == {'aime_2024':30, 'aime_2025':30, 'aime_2026':30, 'amc_2023':40, 'math500':500}
output = Path(c['run']['output_dir'])
assert not (output / 'summary.json').exists(), 'Evaluation output already complete'
assert not any((output / 'rollouts').glob('*.json')), 'Evaluation output already in progress'
output.mkdir(parents=True, exist_ok=True)
(output / 'resolved_config.yaml').write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
(output / 'launch_manifest.json').write_text(json.dumps({
    'datasets': counts, 'rollouts': sum(counts.values()) * 32,
    'gpu_ids': c['actor']['gpu_ids'], 'adapter': str(adapter),
    'gpu_hours_initial': 4,
    'estimate_basis': '20160 responses; rough mean 2000 tokens and 4 GPUs at 800 tokens/s/GPU: 3.5h plus startup. Recalibrate from completed batches.',
}, indent=2))
print('REMOTE PREFLIGHT PASS', counts, flush=True)
PY

# Recheck immediately before creating GPU processes. Never displace a job.
"$python_bin" - <<'PY'
import csv, subprocess
def query(command):
    return list(csv.reader(subprocess.check_output(command, text=True).strip().splitlines()))
gpus = query(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'])
apps = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
occupied = {r[0].strip() for r in apps if r}
for index, uuid, used in gpus:
    if int(index) in [0,1,2,3]:
        assert uuid.strip() not in occupied and int(used) < 1024, f'GPU {index} is occupied'
print('GPU 0,1,2,3 verified empty', flush=True)
PY

export GPU_OWNER=heqiangqiang GPU_TASK=Logic_RL GPU_HOURS=4 GPU_KIND=eval
export GPU_RUN=klopsd8b-step50-avg32-5bench-20261107
export GPU_LOG="$project/outputs/.logs/$GPU_RUN.log"
export FRUGALOPL_PYTHON="$python_bin" FRUGALOPL_TMUX_SESSION="$GPU_RUN"
cache_root=/data/heqiangqiang/.cache/frugalopl-eval
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export TORCH_EXTENSIONS_DIR="$cache_root/torch_extensions"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export HF_HOME="$cache_root/huggingface"
export CUDA_CACHE_PATH="$cache_root/cuda" TORCH_HOME="$cache_root/torch"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
    "$VLLM_CACHE_ROOT" "$HF_HOME" "$CUDA_CACHE_PATH" "$TORCH_HOME"
unset CUDA_VISIBLE_DEVICES DS_ACCELERATOR
bash scripts/start_eval.sh "$config"
