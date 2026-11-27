"""A frozen external Teacher resident on the Learner's own device."""
from __future__ import annotations

from collections.abc import Mapping
import copy
import os

import torch

from utils.algorithm_runtime import apply_generation_chat_template
from utils.policy_model import model_output_head


def external_teacher_config(config):
    learner = config.get('learner', {})
    raw = learner.get('external_teacher') if isinstance(learner, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or type(raw.get('enabled', False)) is not bool:
        raise ValueError('learner.external_teacher.enabled must be boolean.')
    if not raw.get('enabled', False):
        return None
    result = copy.deepcopy(dict(raw))
    for key in ('model_name', 'model_name_or_path'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f'learner.external_teacher.{key} must be a non-empty string.')
    if result.get('enable_thinking', False) is not False:
        raise ValueError('External FKL requires external Teacher No thinking mode.')
    if type(result.get('trust_remote_code', False)) is not bool:
        raise ValueError('external_teacher.trust_remote_code must be boolean.')
    if result.get('dtype', 'bfloat16') not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('External Teacher dtype must be bfloat16, float16 or float32.')
    expected = config.get('algorithm', {}).get('teacher_model_name')
    if expected != result['model_name']:
        raise ValueError('algorithm.teacher_model_name must match the external Teacher label.')
    return result


def validate_teacher_messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError('External Teacher messages must be a non-empty conversation.')
    if not all(isinstance(turn, Mapping) and set(turn) == {'role', 'content'}
               and isinstance(turn['content'], str) and turn['content'].strip() for turn in messages):
        raise ValueError('External Teacher messages require non-empty role/content text.')
    roles = [turn['role'] for turn in messages]
    if roles not in (['user'], ['user', 'assistant', 'user']):
        raise ValueError('External Teacher context must be a question or solution one-shot.')
    if len(messages) == 3 and messages[0]['content'] != messages[2]['content']:
        raise ValueError('Privileged external Teacher must repeat the same question.')
    return [dict(turn) for turn in messages]


def validate_token_alignment(student_tokenizer, teacher_tokenizer, student_model, teacher_model):
    """Vocabulary identity is required before reusing Student response IDs."""
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if not student_vocab or student_vocab != teacher_vocab:
        raise ValueError('External FKL requires identical Student/Teacher token-to-ID vocabularies.')
    student_head = model_output_head(student_model, worker='Student')
    teacher_head = model_output_head(teacher_model, worker='External Teacher')
    student_size = getattr(student_head, 'out_features', None)
    teacher_size = getattr(teacher_head, 'out_features', None)
    if not isinstance(student_size, int) or student_size != teacher_size:
        raise ValueError('External FKL requires identical Student/Teacher output vocabulary sizes.')
    if min(student_vocab.values()) < 0 or max(student_vocab.values()) >= student_size:
        raise ValueError('Tokenizer vocabulary exceeds model output embeddings.')
    for key in ('eos_token_id', 'bos_token_id', 'pad_token_id'):
        if getattr(student_tokenizer, key, None) != getattr(teacher_tokenizer, key, None):
            raise ValueError(f'External FKL requires compatible {key}.')
    return student_size


class FrozenExternalTeacher:
    def __init__(self, config, device, max_prompt_length):
        self.config = dict(config)
        self.device = torch.device(device)
        self.max_prompt_length = max_prompt_length
        self.model = None
        self.tokenizer = None
        self.vocab_size = None

    def initialize(self, student_tokenizer, student_model):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        source = os.path.expanduser(self.config['model_name_or_path'])
        trust = self.config.get('trust_remote_code', False)
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust)
        model = AutoModelForCausalLM.from_pretrained(source,
            torch_dtype=getattr(torch, self.config.get('dtype', 'bfloat16')),
            trust_remote_code=trust, attn_implementation='sdpa')
        model.config.use_cache = False
        model.requires_grad_(False)
        model.eval()
        self.vocab_size = validate_token_alignment(student_tokenizer, tokenizer, student_model, model)
        resident = sum(p.numel()*p.element_size() for p in model.parameters()) / 1024**3
        print(f'[External Teacher] Loading frozen {self.config["model_name"]} on Learner device; '
              f'model parameters={resident:.2f} GiB, shared vocabulary={self.vocab_size}.', flush=True)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def _encode(self, messages):
        prompt = apply_generation_chat_template(self.tokenizer, messages, enable_thinking=False)
        ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError('External Teacher prompt may not be empty.')
        return prompt, ids

    def prompt_ids(self, messages):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('External Teacher has not been initialized.')
        messages = validate_teacher_messages(messages)
        prompt, ids = self._encode(messages)
        limit = self.max_prompt_length
        if limit is None or len(ids) <= limit:
            return prompt, torch.tensor(ids, dtype=torch.long)
        if len(messages) != 3:
            raise ValueError('External Teacher question exceeds max_prompt_length; refusing to truncate it.')
        solution = messages[1]['content']
        messages[1]['content'] = ''
        best_prompt, best_ids = self._encode(messages)
        if len(best_ids) > limit:
            raise ValueError('External Teacher question/template exceeds max_prompt_length; refusing to truncate it.')
        low, high = 0, len(solution)
        while low <= high:
            middle = (low+high)//2
            messages[1]['content'] = solution[:middle]
            candidate, candidate_ids = self._encode(messages)
            if len(candidate_ids) <= limit:
                best_prompt, best_ids = candidate, candidate_ids
                low = middle+1
            else:
                high = middle-1
        return best_prompt, torch.tensor(best_ids, dtype=torch.long)

    def close(self):
        self.model = None
        self.tokenizer = None
        self.vocab_size = None
