"""Sequential TCPO_v1 weight trials using the same foreground Controller."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from collections.abc import Callable, Mapping

import yaml
from utils.eval import write_json
from utils.tcdo_tuning import tuning_decision


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def run_campaign(config: Mapping, *, run_trial: Callable[[dict], object]) -> dict:
    """Restart each changed weight pair from the same seeded initial policy.

    ``run_trial`` returns only after Controller.run() has closed its Workers
    and SwanLab experiment, so consecutive trials never share live GPUs.
    """
    base = copy.deepcopy(dict(config))
    options = base['run'].pop('tcpo_v1_tuning')
    limit = options.get('max_trials', 6)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('run.tcpo_v1_tuning.max_trials must be a positive integer.')
    if base['algorithm']['plugin'] not in {'TCPOV1Plugin', 'TCDOPlugin'}:
        raise ValueError('TCPO_v1 tuning requires TCPOV1Plugin.')
    name = base['run']['name']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('A TCPO_v1 campaign name must be one safe path component.')
    output_root = Path(base['run']['storage']['output_root']).expanduser().resolve()
    campaign_root = output_root / (name + '_campaign')
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_root / 'manifest.json'
    signature = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'algorithm': 'TCPO_v1', 'configuration_sha256': signature, 'trials': [],
    }
    if state['configuration_sha256'] != signature:
        raise ValueError('Campaign configuration changed; use a new run.name for a new experiment.')
    if state.get('status') == 'complete':
        return state
    target_steps = int(base['controller']['max_steps'])
    rule = base['controller'].get('evaluation_stop')
    if not isinstance(rule, Mapping) or rule.get('dataset') != 'aime24' or rule.get('metric') != 'truncation_rate' or rule.get('above') != .5:
        raise ValueError('TCPO_v1 tuning requires the AIME24 truncation stop above 0.5.')
    min_eval_step = int(rule.get('min_step', 10))
    for index in range(len(state['trials']), limit):
        previous = state['trials'][-1]['decision'] if state['trials'] else {
            'incorrect_weight': base['algorithm']['tcdo_incorrect_weight'],
            'truncated_weight': base['algorithm']['tcdo_truncated_weight'],
        }
        incorrect, truncated = previous['incorrect_weight'], previous['truncated_weight']
        trial = copy.deepcopy(base)
        trial_name = f'{name}_r{index:02d}_i{incorrect:g}_t{truncated:g}'
        trial['run']['name'] = trial_name
        trial['run']['redis']['key_prefix'] = base['run']['redis']['key_prefix'] + f'_r{index:02d}'
        trial['algorithm'].update(tcdo_incorrect_weight=incorrect, tcdo_truncated_weight=truncated)
        trial['actor']['evaluation']['before_training'] = index == 0
        run_root = output_root / trial_name
        assert run_root.resolve().is_relative_to(output_root)
        generated = campaign_root / 'configs' / f'trial_{index:03d}.yaml'
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(yaml.safe_dump(trial, sort_keys=False, allow_unicode=True))
        state.update(status='running', active_trial=index, active_run=trial_name,
                     active_config=str(generated), weights={'incorrect': incorrect, 'truncated': truncated})
        _save(manifest_path, state)
        # New trials get their own cloud identity; native resume uses the ID
        # already bound to this trial's output directory.
        os.environ.pop('SWANLAB_RUN_ID', None)
        summary_path = run_root / 'training_summary.json'
        finished = (run_root / 'run_complete.json').exists() or (run_root / 'run_stopped.json').exists()
        if not finished:
            try:
                run_trial(trial)
            except BaseException as error:
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                             error=type(error).__name__)
                _save(manifest_path, state)
                raise
        summary = json.loads(summary_path.read_text())
        initial = run_root / 'checkpoints/step_0/adapter_lora/adapter_model.safetensors'
        initial_hash = _digest(initial) if initial.is_file() else None
        if initial_hash is not None:
            if state.get('initial_adapter_sha256') not in {None, initial_hash}:
                raise RuntimeError('Initial LoRA changed across TCPO_v1 weight trials.')
            state['initial_adapter_sha256'] = initial_hash
        if index == 0:
            state['baseline'] = next((e for e in summary['evaluations'] if e['step'] == 0), None)
        decision = tuning_decision(summary['evaluations'], summary['completed_steps'], incorrect, truncated,
            summary.get('evaluation_stop_reason'), target_steps=target_steps, min_eval_step=min_eval_step)
        if decision['action'] == 'incomplete':
            raise RuntimeError('Trial ended without the required steps or truncation stop.')
        state['trials'].append({'index': index, 'run': trial_name,
            'weights': {'incorrect': incorrect, 'truncated': truncated},
            'completed_steps': summary['completed_steps'], 'evaluations': summary['evaluations'],
            'stop_reason': summary.get('evaluation_stop_reason'), 'decision': decision,
            'initial_adapter_sha256': initial_hash})
        state.update(status='complete' if decision['action'] == 'accept' else 'running')
        _save(manifest_path, state)
        # All evaluation LoRAs and responses remain. Completed trials no
        # longer need optimizer publications or the initial optimizer state.
        for disposable in [run_root / 'tmp/_tmp_lora', run_root / 'checkpoints/step_0/state']:
            assert disposable.resolve().is_relative_to(run_root.resolve())
            if disposable.is_dir():
                shutil.rmtree(disposable)
        print('[TCPO_v1] Trial decision: ' + json.dumps(state['trials'][-1]), flush=True)
        if state['status'] == 'complete':
            return state
    state['status'] = 'review_window_complete'
    _save(manifest_path, state)
    print('[TCPO_v1] Trial budget reached; increase max_trials to continue the saved campaign.', flush=True)
    return state
