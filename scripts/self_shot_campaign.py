"""Run the six authorized self-shot experiments serially and check them hourly.

Run from the owner's project directory on RTX5000-3. Only the explicitly
selected GPUs and processes carrying this campaign's GPU_RUN are inspected
or cleaned up; an occupied card is always waited for, never reclaimed.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

GPU_IDS = [0, 1, 2, 3]
OWNER, TASK = 'heqiangqiang', 'Logic_RL'
RUNS = [(model, mode, hours) for model, hours in [('8b', 4), ('4b', 3), ('1p7b', 2)]
        for mode in ['prompt', 'conversation']]


def emit(event, **values):
    print(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'event': event, **values}, ensure_ascii=False), flush=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def query_gpus(gpu_ids=None):
    selected = GPU_IDS if gpu_ids is None else gpu_ids
    def query(fields, kind):
        raw = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True)
        return list(csv.reader(raw.strip().splitlines()))
    rows = query('index,uuid,memory.used,utilization.gpu', 'gpu')
    cards = {int(row[0]): {'uuid': row[1].strip(), 'memory_mib': int(row[2]),
                         'utilization_percent': int(row[3]), 'processes': []}
             for row in rows if int(row[0]) in selected}
    for uuid, pid in query('gpu_uuid,pid', 'compute-apps'):
        for card in cards.values():
            if card['uuid'] == uuid.strip():
                card['processes'].append(int(pid))
    if set(cards) != set(selected):
        raise RuntimeError('Selected GPU inventory is incomplete')
    return cards


def cards_free(cards):
    return all(not card['processes'] and card['memory_mib'] < 1024 for card in cards.values())


def config_on_devices(config, devices):
    assert len(devices) == len(set(devices)) == 4 and all(isinstance(i, int) and i >= 0 for i in devices)
    result = copy.deepcopy(config)
    for component in ['actor', 'learner']:
        result[component]['gpu_ids'] = list(devices)
    teacher = result.get('worker_defaults', {}).get('Teacher')
    if teacher is not None:
        teacher['gpu_ids'] = list(devices)
    return result


def registration(pid):
    try:
        raw = Path(f'/proc/{pid}/environ').read_bytes()
    except (FileNotFoundError, PermissionError):
        return {}
    values = dict(item.split(b'=', 1) for item in raw.split(b'\0') if b'=' in item)
    return {key: values.get(key.encode(), b'').decode(errors='replace')
            for key in ['GPU_OWNER', 'GPU_TASK', 'GPU_HOURS', 'GPU_KIND', 'GPU_LOG', 'GPU_RUN']}


def inspect_registration(cards, run, hours, log):
    expected = dict(GPU_OWNER=OWNER, GPU_TASK=TASK, GPU_HOURS=str(hours), GPU_KIND='train',
                    GPU_LOG=str(log), GPU_RUN=run)
    actual = {}
    for card in cards.values():
        for pid in card['processes']:
            values = registration(pid)
            # A worker can exit between the NVML query and the procfs read.
            # A vanished PID is not an ownership mismatch.
            if not values and not Path(f'/proc/{pid}').exists():
                continue
            actual[pid] = values
    return actual, bool(actual) and all(values == expected for values in actual.values())


def completed_run(output, expected_steps=100):
    final = read_json(output / 'run_complete.json')
    adapter = output / f'checkpoints/step_{expected_steps}/adapter_lora'
    return bool(final and final.get('completed_steps') == expected_steps
                and (adapter / 'adapter_config.json').is_file()
                and (adapter / 'adapter_model.safetensors').is_file())


def stage_4b_model(project):
    source = Path('/mnt/ystrain-tx-bj/heqiangqiang/models/Qwen3-4B')
    target = Path('/data/heqiangqiang/models/Qwen3-4B-nvme-selfshot-20261107')
    manifest = target / 'self_shot_copy_manifest.json'
    existing = read_json(manifest)
    if existing and all((target / name).is_file() and (target / name).stat().st_size == record['bytes']
                        for name, record in existing.items()):
        return
    target.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    names = set(index['weight_map'].values())
    names.update(path.name for path in source.iterdir() if path.is_file() and path.suffix != '.safetensors')
    records = {}
    for name in sorted(names):
        src, dst = source / name, target / name
        temporary = dst.with_name(dst.name + '.copying')
        digest = hashlib.sha256()
        with src.open('rb') as incoming, temporary.open('wb') as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024**2), b''):
                digest.update(block)
                outgoing.write(block)
        assert temporary.stat().st_size == src.stat().st_size
        temporary.replace(dst)
        records[name] = {'bytes': dst.stat().st_size, 'sha256': digest.hexdigest()}
        emit('model_copy_file', file=name, bytes=records[name]['bytes'])
    atomic_json(manifest, records)
    emit('model_4b_ready', path=str(target))


def monitor_snapshot(output, run, hours, log, previous_step):
    summary = read_json(output / 'training_summary.json', {})
    step = int(summary.get('completed_steps', 0))
    cards = query_gpus()
    actual, valid = inspect_registration(cards, run, hours, log)
    lines = log.read_text(errors='replace').splitlines() if log.exists() else []
    progress = []
    for line in lines:
        if line.startswith('GPU_PROGRESS '):
            try:
                progress.append(json.loads(line[len('GPU_PROGRESS '):]))
            except json.JSONDecodeError:
                # tee may still be appending the newest line during a read.
                continue
    status = ('progressing' if step > previous_step else 'check_evaluation_or_stall')
    snapshot = {'run': run, 'completed_steps': step, 'total_steps': 100,
        'evaluation_count': len(summary.get('evaluations', [])), 'status': status,
        'gpu_registration_valid': valid, 'gpu_processes': actual, 'gpus': cards,
        'latest_progress': progress[-1] if progress else None,
        'last_log_lines': lines[-12:], 'checked_at': time.time()}
    emit('hourly_check', **snapshot)
    return snapshot


def release_finished_run(run, poll_seconds):
    """Release only proven own orphaned CUDA processes after successful exit."""
    started = time.monotonic()
    while True:
        cards = query_gpus()
        if cards_free(cards):
            return
        own = []
        for card in cards.values():
            for pid in card['processes']:
                values = registration(pid)
                if (values.get('GPU_OWNER') == OWNER and values.get('GPU_TASK') == TASK
                        and values.get('GPU_RUN') == run):
                    own.append(pid)
        if not own:
            # This run has released its contexts. A later job may have taken
            # the cards; the next launch still performs its own empty-card check.
            return
        if time.monotonic() - started > 120:
            for card in cards.values():
                for pid in card['processes']:
                    values = registration(pid)
                    if (values.get('GPU_OWNER') == OWNER and values.get('GPU_TASK') == TASK
                            and values.get('GPU_RUN') == run):
                        emit('cleanup_completed_own_process', run=run, pid=pid)
                        try:
                            os.kill(pid, 15)
                        except ProcessLookupError:
                            pass
            started = time.monotonic()
        time.sleep(poll_seconds)


def main(args):
    global GPU_IDS
    GPU_IDS = [int(value.strip()) for value in args.gpu_devices.split(',')]
    assert len(GPU_IDS) == len(set(GPU_IDS)) == 4 and min(GPU_IDS) >= 0
    project = args.project.resolve()
    if project.parent != Path('/home/ubuntu/heqiangqiang'):
        raise RuntimeError('Campaign must run under the owner directory')
    if subprocess.check_output(['hostname'], text=True).strip() != 'VM-8-245-ubuntu':
        raise RuntimeError('Only RTX5000-3 is authorized')
    os.chdir(project)
    sys.path.insert(0, str(project))
    from engine.actor import load_merged_config
    state_dir = project / 'outputs/campaign'
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = state_dir / 'state.json'
    # The copy uses no GPU; overlap it with the first 8B run's startup.
    copy_errors = []
    def copy_model():
        try:
            stage_4b_model(project)
        except Exception as error:
            copy_errors.append(repr(error))
            emit('model_copy_failed', error=repr(error))
    copy_thread = threading.Thread(target=copy_model, daemon=True)
    copy_thread.start()

    configurations = []
    for model, mode, hours in RUNS:
        path = project / f'configs/training/self_shot_opsd/qwen3_{model}_{mode}_4GPU.yaml'
        config = config_on_devices(load_merged_config(path), GPU_IDS)
        runtime_path = state_dir / f'runtime_config_{model}_{mode}.json'
        atomic_json(runtime_path, config)
        assert config['algorithm']['enable_thinking'] is False
        assert config['algorithm']['self_shot_format'] == mode
        assert config['controller']['max_steps'] == 100 and config['eval_interval'] == 10
        assert config['actor']['evaluation']['rollouts_per_prompt'] == 8
        assert config['actor']['gpu_ids'] == config['learner']['gpu_ids'] == GPU_IDS
        configurations.append((model, mode, hours, runtime_path, config))

    previous_eval = Path('/data/heqiangqiang/outputs/FrugalOPL/kl_opsd_qwen3_8b_step50_avg32_5bench_20261107')
    previous_status = Path('/home/ubuntu/heqiangqiang/FrugalOPL-eval8b-step50-avg32-20261107/outputs/.logs/klopsd8b-step50-avg32-5bench-20261107.status')
    while not (previous_status.exists() and previous_status.read_text().strip() == '0'
               and (previous_eval / 'summary.json').is_file() and (previous_eval / 'details.json').is_file()):
        if previous_status.exists() and previous_status.read_text().strip() != '0':
            raise RuntimeError('Preceding evaluation failed; refusing to hide it by starting training')
        atomic_json(state_path, {'phase': 'waiting_for_previous_evaluation', 'updated_at': time.time()})
        time.sleep(args.poll_seconds)

    for index, (model, mode, hours, path, config) in enumerate(configurations):
        run = config['run']['name']
        session = f'self-shot-{model}-{mode}-20261107'
        log = project / f'outputs/.logs/{session}.log'
        status = log.with_suffix('.status')
        output = Path(config['run']['storage']['output_root']) / run
        if completed_run(output):
            emit('already_complete', run=run)
            continue
        if model == '4b':
            copy_thread.join()
            if copy_errors:
                raise RuntimeError(f'4B model staging failed: {copy_errors}')
        assert Path(config['run']['policy']['model_name_or_path'], 'config.json').is_file()
        active = subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0
        if not active:
            if status.exists() or (output / 'training_summary.json').exists():
                raise RuntimeError(f'Incomplete earlier attempt requires review, not silent overwrite: {run}')
            while not cards_free(query_gpus()):
                atomic_json(state_path, {'phase': 'waiting_for_free_gpus', 'index': index, 'run': run, 'gpu_ids': GPU_IDS,
                                         'updated_at': time.time()})
                time.sleep(args.poll_seconds)
            env = dict(os.environ, GPU_OWNER=OWNER, GPU_TASK=TASK, GPU_HOURS=str(hours), GPU_KIND='train',
                GPU_LOG=str(log), GPU_RUN=run, FRUGALOPL_TMUX_SESSION=session,
                FRUGALOPL_PYTHON=sys.executable,
                FRUGALOPL_SWANLAB_ENV_FILE='/home/ubuntu/heqiangqiang/.config/frugalopl/swanlab.env')
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.pop('DS_ACCELERATOR', None)
            subprocess.run(['bash', 'scripts/start_train.sh', str(path)], env=env, check=True)
            emit('training_started', run=run, model=model, format=mode, log=str(log), gpu_hours=hours)
        saved_checks = sorted(state_dir.glob(f'hourly_{index}_*.json'))
        last_snapshot = read_json(saved_checks[-1], {}) if saved_checks else {}
        merged_config = output / 'merged_config.json'
        first_started = merged_config.stat().st_mtime if merged_config.exists() else time.time()
        monitor_epoch = last_snapshot.get('checked_at', first_started)
        last_hour = time.monotonic() - max(0, time.time() - monitor_epoch)
        previous_step, last_registration = int(last_snapshot.get('completed_steps', 0)), None
        while True:
            atomic_json(state_path, {'phase': 'training', 'index': index, 'total_runs': 6, 'run': run, 'gpu_ids': GPU_IDS,
                'log': str(log), 'output': str(output), 'updated_at': time.time()})
            if completed_run(output):
                # The final metrics and checkpoint are committed before GPU
                # teardown / telemetry. New users taking already released
                # cards must not turn a finished run into a queue failure.
                emit('training_complete', run=run, completed_steps=100)
                break
            # Completion wins over ownership inspection: another legitimate
            # job may already have taken a card released by this finished run.
            if status.exists():
                code = int(status.read_text().strip())
                if code != 0 or not completed_run(output):
                    raise RuntimeError(f'Run failed or lacks step-100 completion/checkpoint: {run}, exit={code}')
                emit('training_complete', run=run, completed_steps=100)
                break
            cards = query_gpus()
            actual, valid = inspect_registration(cards, run, hours, log)
            if actual and actual != last_registration:
                emit('actual_gpu_registration', run=run, valid=valid, processes=actual)
                last_registration = actual
                if not valid:
                    raise RuntimeError('Selected GPUs contain an unregistered or unrelated process; manual review required')
            if time.monotonic() - last_hour >= args.monitor_seconds:
                snapshot = monitor_snapshot(output, run, hours, log, previous_step)
                atomic_json(state_dir / f'hourly_{index}_{int(time.time())}.json', snapshot)
                previous_step, last_hour = snapshot['completed_steps'], time.monotonic()
            time.sleep(args.poll_seconds)
        # A completed Controller must also have released its CUDA contexts.
        release_finished_run(run, args.poll_seconds)
    atomic_json(state_path, {'phase': 'complete', 'completed_runs': 6, 'gpu_ids': GPU_IDS, 'updated_at': time.time()})
    emit('campaign_complete', completed_runs=6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--monitor-seconds', type=float, default=3600)
    parser.add_argument('--gpu-devices', default='0,1,2,3')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        emit('campaign_failed', error=repr(error))
        atomic_json(arguments.project / 'outputs/campaign/state.json', {
            'phase': 'failed', 'error': repr(error), 'updated_at': time.time()})
        raise
