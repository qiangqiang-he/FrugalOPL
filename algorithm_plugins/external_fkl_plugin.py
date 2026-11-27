"""External-model on-policy Forward KL, with optional solution one-shot."""
from collections.abc import Mapping
from dataclasses import replace

import torch

from utils.algorithm_runtime import controller_questions_and_answers
from utils.prompts import render_prompt_from_record
from utils.tensor_signals import masked_token_mean
from .algorithm_base import ObjectiveResult
from .kl_opsd_plugin import KLOPSDPlugin
from .pg_opsd_plugin import PGOPSDPlugin
from .plugin_registry import register_plugin_class


@register_plugin_class
class ExternalFKLPlugin(KLOPSDPlugin):
    requires_frozen_reference = False
    requires_external_teacher = True
    external_context = 'question'
    external_algorithm = 'FKL_external'

    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('kl_token_clip') is not None:
            raise ValueError('External FKL requires kl_token_clip: null.')
        runtime = replace(runtime, algorithm={**runtime.algorithm, 'kl_token_clip': None})
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        self._teacher_model_name = runtime.algorithm.get('teacher_model_name')
        if not isinstance(self._teacher_model_name, str) or not self._teacher_model_name.strip():
            raise ValueError('External FKL requires algorithm.teacher_model_name.')
        if self._enable_thinking:
            raise ValueError('External FKL requires Student No thinking mode.')
        if runtime.algorithm.get('teacher_sync_interval', -1) != -1:
            raise ValueError('External FKL requires a fixed external Teacher.')

    @staticmethod
    def _training_records(signals):
        questions, _ = controller_questions_and_answers(signals)
        records = signals.get('records')
        if (not isinstance(records, list) or len(records) != len(questions)
                or not all(isinstance(record, Mapping) and isinstance(record.get('question'), str)
                           and record['question'].strip() for record in records)):
            raise ValueError('External FKL requires question-aligned training records.')
        return records

    def _teacher_messages(self, record):
        return [{'role': 'user', 'content': render_prompt_from_record(self._train_template, record)}]

    def collect_learner_signals(self, signals):
        records = self._training_records(signals)
        indices = signals.get('traj', {}).get('prompt_indices')
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('External FKL requires one-dimensional integer prompt_indices.')
        order = indices.tolist()
        if any(i < 0 or i >= len(records) for i in order):
            raise ValueError('External FKL prompt index is out of range.')
        return {'teacher_messages': [self._teacher_messages(records[i]) for i in order]}

    def compute_objective(self, batch):
        values = self.compute_external_forward_kl(batch, teacher_messages=batch.get('teacher_messages'),
                                                  temperature=self._kl_temperature)
        mask = batch['response_mask'].bool()
        loss = masked_token_mean(values['response_forward_kl'], mask)
        metrics = {'external_fkl_loss': loss.detach(),
            'external_fkl_unclipped_kl': masked_token_mean(values['response_forward_kl_unclipped'], mask).detach(),
            'external_fkl_clip_fraction': masked_token_mean(values['response_forward_kl_clipped_fraction'], mask).detach(),
            'student_entropy_nats': masked_token_mean(values['response_student_entropy_nats'], mask).detach(),
            'teacher_entropy_nats': masked_token_mean(values['response_teacher_entropy_nats'], mask).detach()}
        return ObjectiveResult(-loss, metrics, mask.sum().detach(), frozenset(metrics))

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': self.external_algorithm,
                'teacher_model_name': self._teacher_model_name, 'teacher_context': self.external_context}


@register_plugin_class
class ExternalPrivilegedFKLPlugin(ExternalFKLPlugin):
    external_context = 'solution_one_shot'
    external_algorithm = 'OPD_external_self_shot'

    @staticmethod
    def _training_records(signals):
        return PGOPSDPlugin._training_records(signals)

    def _teacher_messages(self, record):
        question = render_prompt_from_record(self._train_template, record)
        return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': record['solution']},
                {'role': 'user', 'content': question}]
