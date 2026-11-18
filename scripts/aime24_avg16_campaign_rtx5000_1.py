"""Durably run all three AIME24 conditions/model on every currently free GPU."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/heqiangqiang/miniforge3/envs/rlvr/bin/python'
CONFIGS = ROOT/'configs/eval/aime24_avg16_20261108'
SIZES = ['4B', '1.7B', '8B']


def event(phase, **values):
    print(json.dumps({'phase': phase, 'updated_at': int(time.time()), **values}), flush=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def free_gpus():
    def query(kind, fields):
        text = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                        '--format=csv,noheader,nounits'], text=True)
        return [[s.strip() for s in row] for row in csv.reader(text.splitlines()) if row]
    gpus = query('gpu', 'index,uuid,memory.used')
    occupied = {row[0] for row in query('compute-apps', 'gpu_uuid,pid')}
    return [int(i) for i,u,m in gpus if u not in occupied and int(m) < 1024]


def model_ready(size):
    model = Path(f'/data/heqiangqiang/cache/models/Qwen3-{size}')
    if not all((model/name).is_file() for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']):
        return False
    if size != '4B' and not (model/'download_manifest.json').is_file():
        status = ROOT/f'outputs/.logs/download-{size}.status'
        if status.exists() and status.read_text().strip() != '0':
            raise RuntimeError(f'Model download failed: {size}; inspect download-{size}.log')
        return False
    return True


def verify_results(size):
    out = ROOT/f'outputs/Qwen3-{size}'
    summary = json.loads((out/'summary.json').read_text())
    details = json.loads((out/'details.json').read_text())
    assert len(details['datasets']) == 3
    for ds in details['datasets'].values():
        assert len(ds['questions']) == 30
        for q in ds['questions']:
            assert len(q['rollouts']) == 16
            assert all(isinstance(r.get('response'), str) and r['response'] for r in q['rollouts'])
    assert summary['overall']['response_count'] == 1440
    write(out/'completion_verified.json', {'verified': True, 'response_count': 1440,
                                          'updated_at': int(time.time())})
    return {ds: v['metrics']['avg_at_16'] for ds,v in summary['datasets'].items()}


def registered_run_alive(run):
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            env = dict(item.split('=',1) for item in (path/'environ').read_bytes().decode().split('\0') if '=' in item)
        except (OSError,UnicodeDecodeError):
            continue
        if env.get('GPU_OWNER') == 'heqiangqiang' and env.get('GPU_RUN') == run:
            return True
    return False


def launch(size):
    output = ROOT/f'outputs/Qwen3-{size}'
    previous_manifest = output/'launch_manifest.json'
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.is_file() else None
    if previous:
        registration = previous['registration']
        assert registration['GPU_OWNER'] == 'heqiangqiang' and registration['GPU_KIND'] == 'eval'
        old_run = previous['run']
        if subprocess.run(['tmux','has-session','-t','='+old_run],capture_output=True).returncode == 0:
            event('reattached',size=size,run=old_run)
            return old_run
        if registered_run_alive(old_run):
            event('waiting_for_owned_process_exit',size=size,run=old_run)
            return None
    has_partial = any((output/'rollouts').glob('batch_*.json'))
    if has_partial:
        assert previous is not None, 'Partial output without an ownership manifest needs review'
        assert output.resolve().parent == (ROOT/'outputs').resolve()
        archive = ROOT/'outputs/retries'/f'Qwen3-{size}-{time.time_ns()}'
        archive.parent.mkdir(parents=True,exist_ok=True)
        output.rename(archive)
        event('preserved_failed_attempt',size=size,path=str(archive))
    model = Path(f'/data/heqiangqiang/cache/models/Qwen3-{size}')
    inspected = subprocess.check_output([PYTHON, 'tests/helpers/inspect_qwen_models_remote.py', str(model)],
                                        cwd=ROOT, text=True)
    assert json.loads(inspected.strip().splitlines()[-1])['valid'], inspected
    selected = free_gpus()
    if not selected:
        return None
    cfg = json.loads((CONFIGS/f'Qwen3-{size}.yaml').read_text())
    cfg['actor']['gpu_ids'] = selected
    run = f'aime24-avg16-qwen3-{size.replace(".", "p").lower()}-20261108'
    if previous or any((ROOT/'outputs/retries').glob(f'Qwen3-{size}-*')):
        run += f'-retry-{time.time_ns()}'
    runtime = ROOT/f'outputs/runtime-{size}.json'
    output = ROOT/f'outputs/Qwen3-{size}'
    assert not (output/'summary.json').exists(), f'Completed output already exists: {size}'
    assert not any((output/'rollouts').glob('batch_*.json')), f'Partial run needs review: {size}'
    write(runtime, cfg)
    env = os.environ.copy()
    conda = str(Path(PYTHON).parent.parent)
    cache = Path('/data/heqiangqiang/cache/aime24-avg16-20261108')
    env.update(CONDA_PREFIX=conda, PATH=f'{conda}/bin:'+env.get('PATH',''),
        CUDA_HOME=conda, LD_LIBRARY_PATH=f'{conda}/lib:'+env.get('LD_LIBRARY_PATH',''),
        GPU_OWNER='heqiangqiang', GPU_TASK='logic RL', GPU_HOURS='2' if size == '8B' else '1',
        GPU_KIND='eval', GPU_RUN=run, GPU_LOG=str(ROOT/f'outputs/.logs/{run}.log'),
        FRUGALOPL_PYTHON=PYTHON, FRUGALOPL_TMUX_SESSION=run,
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for key, subdir in {'TRITON_CACHE_DIR':'triton', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TORCH_EXTENSIONS_DIR':'extensions', 'VLLM_CACHE_ROOT':'vllm', 'HF_HOME':'huggingface',
        'CUDA_CACHE_PATH':'cuda', 'TORCH_HOME':'torch'}.items():
        env[key] = str(cache/subdir)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('DS_ACCELERATOR', None)
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
    subprocess.run([PYTHON, 'tests/helpers/aime24_avg16_preflight.py', '--config', str(runtime),
                    '--tokenizer', str(model), '--output', str(output)], cwd=ROOT, env=cpu_env, check=True)
    # Recheck immediately before the actual GPU-creating launcher.
    if not set(selected).issubset(free_gpus()):
        event('gpu_changed_before_launch', size=size)
        return None
    subprocess.run(['bash', 'scripts/start_eval.sh', str(runtime)], cwd=ROOT, env=env, check=True)
    manifest = {'size': size, 'gpu_ids': selected, 'run': run,
        'registration': {k:env[k] for k in ['GPU_OWNER','GPU_TASK','GPU_HOURS','GPU_KIND','GPU_LOG','GPU_RUN']},
        'config': str(runtime), 'output': str(output), 'started_at': int(time.time())}
    write(output/'launch_manifest.json', manifest)
    event('launched', **manifest)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for size in SIZES:
        cfg = json.loads((CONFIGS/f'Qwen3-{size}.yaml').read_text())
        assert cfg['actor']['evaluation']['rollouts_per_prompt'] == 16
        assert len(cfg['actor']['evaluation']['datasets']) == 3
    if args.check_only:
        event('campaign_configuration_validated', models=SIZES, conditions_per_model=3)
        return
    assert socket.gethostname() == 'VM-8-172-ubuntu'
    os.chdir(ROOT)
    lock = (ROOT/'outputs/campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = {}
    pending = SIZES.copy()
    while pending:
        for size in pending.copy():
            output = ROOT/f'outputs/Qwen3-{size}'
            if (output/'summary.json').is_file() and (output/'details.json').is_file():
                completed[size] = verify_results(size)
                pending.remove(size)
                event('recovered_completed_model',size=size)
        write(ROOT/'outputs/campaign_results.json', {'completed': completed, 'pending': pending})
        if not pending:
            break
        ready = next((s for s in pending if model_ready(s)), None)
        if ready is None:
            event('waiting', model_files_ready=ready, pending=pending)
            time.sleep(30)
            continue
        run = launch(ready)
        if run is None:
            time.sleep(10)
            continue
        status = ROOT/f'outputs/.logs/{run}.status'
        while not status.exists():
            # tmux exit without a status must not silently hang the campaign.
            alive = subprocess.run(['tmux', 'has-session', '-t', '='+run], capture_output=True).returncode == 0
            if not alive:
                time.sleep(2)
                if not status.exists():
                    raise RuntimeError(f'Evaluation exited without status: {run}')
            time.sleep(30)
        assert status.read_text().strip() == '0', f'Evaluation failed: {run}; inspect its GPU_LOG'
        completed[ready] = verify_results(ready)
        pending.remove(ready)
        write(ROOT/'outputs/campaign_results.json', {'completed': completed, 'pending': pending})
        event('model_complete', size=ready, scores=completed[ready], pending=pending)
    event('all_complete', results=completed, free_gpu_ids=free_gpus())
    lock.close()


if __name__ == '__main__':
    main()
