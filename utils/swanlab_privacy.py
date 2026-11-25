"""The explicit public-data boundary for SwanLab uploads.

Keep the runnable configuration and detailed diagnostics on disk. Only the
selected experiment parameters below may enter cloud configuration records.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from functools import wraps
import math
import re


_FIELDS = {
    (): 'eval_interval save_ckpt_interval',
    ('run',): 'seed',
    ('controller',): 'max_steps train_prompt_batch_size max_inflight_batches overlap_actor_learner prefetch_train_during_evaluation prefetch_work_stealing',
    ('controller', 'data'): 'max_prompt_length max_response_length',
    ('run', 'policy'): 'dtype',
    ('run', 'policy', 'lora'): 'enabled rank alpha dropout',
    ('actor', 'engine'): 'tensor_parallel_size enforce_eager max_model_len gpu_memory_utilization max_num_seqs max_num_batched_tokens enable_chunked_prefill kv_cache_memory_bytes',
    ('actor', 'evaluation'): 'enabled before_training distribute_replies work_stealing max_examples rollouts_per_batch rollouts_per_prompt',
    ('actor', 'evaluation', 'generation'): 'temperature top_p top_k max_response_length seed',
    ('learner',): 'max_sequence_length',
    ('learner', 'execution'): 'gradient_checkpointing clip_grad_norm offload_frozen_base_between_turns',
    ('learner', 'optimizer'): 'type lr betas eps weight_decay',
    ('learner', 'scheduler'): 'type',
    ('algorithm',): 'plugin algo_name kl_top_k loss_aggregation enable_thinking self_shot_format teacher_sync_interval kl_temperature kl_token_clip tra_plausible_fraction tra_ratio_power ratio_plausible_fraction tca_plausible_fraction tcdo_plausible_fraction tcdo_incorrect_weight tcdo_truncated_weight tcpo_objective_version tcpo_plausible_fraction tcpo_alpha_plus tcpo_alpha_minus tcpo_alpha_tru',
    ('algorithm', 'training_sampling'): 'temperature top_p top_k max_tokens n seed',
}
_ENUMS = frozenset({'bfloat16', 'float16', 'float32', 'bf16', 'fp16', 'fp32',
                   'token', 'sequence', 'conversation', 'prompt', 'adam', 'adamw',
                   'constant', 'linear', 'cosine'})
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.+ -]{0,159}\Z')
_PRIVATE_KEY = re.compile(r'(?:^|[_./-])(?:path|dir|directory|hostname|username|password|secret|credential|api_key|apikey|access_token|auth_token|url|uri|address|host|ip)(?:$|[_./-])', re.I)
_IP = re.compile(r'(?:\d{1,3}\.){3}\d{1,3}')
_PUBLIC_DATASETS = frozenset({'amc23', 'aime24', 'gsm8k', 'openthoughts_math30k'})
PRIVATE_ERROR = 'Detailed error information is retained in local logs.'


def public_label(value: object, fallback: str = 'unspecified') -> str:
    """Public display labels cannot contain paths, URLs or contact addresses."""
    if isinstance(value, str) and _LABEL.fullmatch(value) and not _IP.search(value):
        return value
    return fallback


def _get(config: Mapping, path: tuple[str, ...]) -> Mapping:
    value = config
    for key in path:
        value = value.get(key, {})
        if not isinstance(value, Mapping):
            return {}
    return value


def _parameter(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, str):
        return value.lower() in _ENUMS
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    return False


def thinking_mode(config: Mapping) -> str:
    run = _get(config, ('run',))
    model = str(run.get('group', '')) + ' ' + str(_get(config, ('run', 'policy')).get('model_name_or_path', ''))
    if 'deepseek' in model.lower() and ('distill' in model.lower() or 'r1' in model.lower()):
        return 'native'
    return 'thinking' if _get(config, ('algorithm',)).get('enable_thinking', True) else 'no-thinking'


def public_run_name(config: Mapping, name: str) -> str:
    """Keep mode labels consistent when a matrix configuration is toggled."""
    title = _get(config, ('algorithm',)).get('algo_name')
    if title in {'FKL', 'Clip-OPSD', 'TRA'}:
        name = re.sub(r'_(?:no-thinking|thinking)(?=_|$)', '', name)
        mode = thinking_mode(config)
        if mode != 'native':
            name = re.sub(r'(?=_\d+steps_)', '_' + mode, name, count=1) if re.search(r'_\d+steps_', name) else name + '_' + mode
    return public_label(name, 'training')


def public_config(config: Mapping) -> dict:
    """Allowlist parameters; never copy storage, prompts, hosts or model paths."""
    result = {'privacy_schema': 1}
    for path, fields in _FIELDS.items():
        source = _get(config, path)
        selected = {key: copy.deepcopy(source[key]) for key in fields.split() if key in source and _parameter(source[key])}
        if not selected:
            continue
        destination = result
        for key in path:
            destination = destination.setdefault(key, {})
        destination.update(selected)
    run = result.setdefault('run', {})
    run['model_name'] = public_label(_get(config, ('run',)).get('group'))
    run['thinking_mode'] = thinking_mode(config)
    algorithm = result.setdefault('algorithm', {})
    for key in ('plugin', 'algo_name'):
        raw = _get(config, ('algorithm',)).get(key)
        if raw is not None:
            algorithm[key] = public_label(raw)
    for component in ('actor', 'learner'):
        ids = _get(config, (component,)).get('gpu_ids')
        if isinstance(ids, (list, tuple)):
            result.setdefault(component, {})['gpu_count'] = len(ids)
    evaluations = _get(config, ('controller', 'data')).get('eval', [])
    if isinstance(evaluations, list):
        result['evaluation_datasets'] = [entry['name'] if isinstance(entry, Mapping) and entry.get('name') in _PUBLIC_DATASETS
                                         else f'dataset_{index+1}' for index, entry in enumerate(evaluations)]
    return result


def public_metric_key(value: str) -> bool:
    return bool(value and not value.startswith(('/', '\\', '~', '.'))
                and re.fullmatch(r'[A-Za-z0-9_./-]+', value)
                and '..' not in value and not _PRIVATE_KEY.search(value) and not _IP.search(value))


def private_error(error: object) -> str | None:
    return None if error is None else PRIVATE_ERROR


def _guard_finish(run):
    if getattr(run, '_frugalopl_private_finish', False):
        return run
    original = run.finish

    @wraps(original)
    def finish(state='success', error=None, async_log_timeout=None):
        options = {'state': state, 'error': private_error(error)}
        if async_log_timeout is not None:
            options['async_log_timeout'] = async_log_timeout
        return original(**options)

    # SDK exception and SIGINT handlers also call this public method. Guarding
    # the instance covers automatic tracebacks as well as Controller.finish().
    run.finish = finish
    run._frugalopl_private_finish = True
    return run


def _setting(settings, *keys):
    value = settings
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
    return value


def init_public_run(swanlab, *, config: Mapping, **options):
    """Configure privacy before SDK initialization, including its auto collectors."""
    name = public_run_name(config, options.pop('name', 'training'))
    group = public_label(options.pop('group', None), 'experiments')
    settings = swanlab.Settings(
        interactive=False,
        terminal={'proxy_type': 'none'},
        probe={'hardware': False, 'runtime': False, 'requirements': False,
               'conda': False, 'git': False, 'swanlab': False, 'monitor': True},
        experiment={'name': name, 'group': group, 'description': 'Public experiment parameters and scalar metrics.', 'tags': [], 'job_type': options.get('job_type', 'training')},
        run={'config': None, 'history': None},
        integration={'webhook': {'url': '', 'value': ''}},
    )
    required = {('terminal', 'proxy_type'): 'none',
                **{('probe', field): False for field in
                   ('hardware', 'runtime', 'requirements', 'conda', 'git', 'swanlab')}}
    if any(_setting(settings, *path) != expected for path, expected in required.items()):
        raise RuntimeError('SwanLab SDK does not support the required public-data privacy controls.')
    if (_setting(settings, 'experiment', 'tags') != []
            or _setting(settings, 'experiment', 'description') != 'Public experiment parameters and scalar metrics.'
            or _setting(settings, 'integration', 'webhook', 'url')):
        raise RuntimeError('SwanLab inherited tags or webhook settings were not cleared.')
    run_settings = settings.run
    history = run_settings.get('history') if isinstance(run_settings, Mapping) else run_settings.history
    if history is not None:
        raise ValueError('External SwanLab history replay is disabled for public logging.')
    was_active = getattr(swanlab, 'has_run', lambda: False)()
    try:
        run = swanlab.init(config=public_config(config), settings=settings,
                           name=name, group=group, **options)
        return _guard_finish(run)
    except BaseException:
        # A partially initialized SDK run must not upload the exception hook's
        # raw traceback during interpreter shutdown either.
        if not was_active and getattr(swanlab, 'has_run', lambda: False)():
            try:
                _guard_finish(swanlab.get_run()).finish(state='crashed', error=PRIVATE_ERROR)
            except Exception:
                pass
        raise
