#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == VM-8-245-ubuntu ]]
project=/home/ubuntu/heqiangqiang/FrugalOPL-self-shot-opsd-20261107
archive=/home/ubuntu/heqiangqiang/frugalopl-self-shot-20261107.tar.gz
python_bin=/data/heqiangqiang/miniforge3/envs/rlvr/bin/python
[[ "$(sha256sum "$archive" | cut -d ' ' -f1)" == "$1" ]]
mkdir -p "$project"
tar -xzf "$archive" -C "$project"
cd "$project"
bash -n scripts/start_train.sh
CUDA_VISIBLE_DEVICES="" DS_ACCELERATOR=cpu "$python_bin" - <<'PY'
import json
from pathlib import Path
from algorithm_plugins import SelfShotOPSDPlugin, plugin_factory_from_config
from engine.actor import load_merged_config
from scripts.self_shot_campaign import RUNS, GPU_IDS
from transformers import AutoTokenizer

old = json.loads(Path('/data/heqiangqiang/outputs/FrugalOPL/kl_opsd_qwen3_8b_math30k_b64n1_aime24_12288_nothinking_20261106/merged_config.resume.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(old['run']['policy']['model_name_or_path'], local_files_only=True)
records = json.loads(Path('data/openthoughts_math30k.json').read_text())
assert isinstance(records, list) and records
print('Training records:', len(records), flush=True)
assert all(isinstance(r.get('solution'), str) and r['solution'].strip() for r in records)
assert not any('<think>' in r['solution'] or '</think>' in r['solution'] for r in records), 'Historical assistant solutions contain thinking delimiters; inspect preservation before launch'
sample = records[0]
for model, mode, hours in RUNS:
    config = load_merged_config(f'configs/training/self_shot_opsd/qwen3_{model}_{mode}_4GPU.yaml')
    assert config['controller'] == old['controller']
    assert config['actor']['engine'] == old['actor']['engine']
    assert config['actor']['evaluation'] == old['actor']['evaluation']
    for key in ['execution', 'optimizer', 'scheduler', 'checkpoint', 'max_sequence_length']:
        assert config['learner'][key] == old['learner'][key]
    numerical = dict(config['algorithm'])
    numerical.pop('self_shot_format')
    numerical.update(plugin='KLOPSDPlugin', algo_name='kl_opsd')
    assert numerical == old['algorithm']
    assert config['run']['policy']['lora'] == old['run']['policy']['lora']
    plugin = plugin_factory_from_config(config)(tokenizer)
    assert isinstance(plugin, SelfShotOPSDPlugin)
    student = plugin.collect_actor_signals({'questions': [sample['question']], 'answers': [sample['answer']], 'records': [sample]})['prompts'][0]
    teacher = plugin._render_privileged_prompt(sample)
    assert sample['solution'] not in student
    assert teacher.count(sample['question']) == 2
    assert student.endswith('<think>\n\n</think>\n\n') and teacher.endswith('<think>\n\n</think>\n\n')
    print('PREFLIGHT', model, mode, 'recipe unchanged; No Thinking; same Q twice; student has no solution', flush=True)
PY

mkdir -p outputs/redis outputs/.logs
CUDA_VISIBLE_DEVICES="" "$python_bin" - <<'PY'
from pathlib import Path
import redis, subprocess, time
directory = Path.cwd() / 'outputs/redis'
client = redis.Redis(host='127.0.0.1', port=16479)
try:
    client.ping()
except redis.ConnectionError:
    subprocess.run(['/data/heqiangqiang/miniforge3/pkgs/redis-server-8.10.2-h89b2107_1/bin/redis-server',
        '--bind', '127.0.0.1', '--port', '16479', '--daemonize', 'yes', '--save', '', '--appendonly', 'no',
        '--dir', str(directory), '--pidfile', str(directory / 'redis.pid'), '--logfile', str(directory / 'redis.log')], check=True)
    for _ in range(20):
        try:
            client.ping()
            break
        except redis.ConnectionError:
            time.sleep(.25)
    else:
        raise RuntimeError('Private Redis did not start')
assert client.config_get('dir')['dir'] == str(directory), 'Port 16479 belongs to a different project; refusing to reuse it'
print('Private Redis ready on 16479', flush=True)
PY

session=self-shot-opsd-six-runs-20261107
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Campaign already running: $session"
    exit 0
fi
campaign_log="$project/outputs/.logs/campaign.log"
printf -v python_command '%q ' "$python_bin" scripts/self_shot_campaign.py --project "$project"
printf -v quoted_log '%q' "$campaign_log"
script="set -o pipefail; source /data/heqiangqiang/miniforge3/etc/profile.d/conda.sh; conda activate rlvr; unset CUDA_VISIBLE_DEVICES DS_ACCELERATOR; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; $python_command 2>&1 | tee -a $quoted_log; exit \${PIPESTATUS[0]}"
printf -v quoted_script '%q' "$script"
tmux new-session -d -s "$session" -c "$project" "bash -lc $quoted_script"
echo "Campaign started: $session"
echo "Campaign log: $campaign_log"
echo "State: $project/outputs/campaign/state.json"
