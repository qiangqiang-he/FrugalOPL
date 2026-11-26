"""Own-rollout answer privilege shared by forward and reverse alignment."""
from collections.abc import Mapping, Sequence

import torch

from utils.answer_verifier import extract_final_answer
from .algorithm_base import ObjectiveResult


class MatchedAnswerTeacherMixin:
    _match_metric_prefix = 'tra'

    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('answer_context', 'single_turn') != 'single_turn':
            raise ValueError('Answer privileges require answer_context: single_turn.')
        fallback = runtime.algorithm.get('match_answer_fallback', 'gold')
        if fallback not in ('gold', 'empty'):
            raise ValueError('match_answer_fallback must be gold or empty.')
        self._match_answer_fallback = fallback
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def _render_teacher_record(self, record):
        return self._render_answer_teacher_record(record, allow_empty=self._match_answer_fallback == 'empty')

    def collect_actor_signals(self, signals):
        request = super().collect_actor_signals(signals)
        request['return_signals'] = {**request['return_signals'], 'responses': True}
        return request

    def collect_learner_signals(self, signals):
        records = self._training_records(signals)
        trajectory = signals.get('traj')
        if not isinstance(trajectory, Mapping):
            raise ValueError('Matched-answer alignment requires a trajectory mapping.')
        indices = trajectory.get('prompt_indices')
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError('Matched-answer prompt_indices must be a one-dimensional integer tensor.')
        indices = indices.tolist()
        if any(index < 0 or index >= len(records) for index in indices):
            raise ValueError('Matched-answer prompt index is outside the current records.')
        responses = trajectory.get('responses', signals.get('responses'))
        if responses is None:
            responses = [''] * len(indices)
        if (isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence)
                or len(responses) != len(indices) or not all(isinstance(value, str) for value in responses)):
            raise ValueError('Matched-answer responses must be rollout-aligned strings.')
        answers, extracted = [], []
        for index, response in zip(indices, responses):
            answer, found = extract_final_answer(response)
            fallback = '' if self._match_answer_fallback == 'empty' else str(records[index]['answer']).strip()
            answers.append(answer if found else fallback)
            extracted.append(found)
        return {
            'teacher_prompts': [self._render_privileged_prompt({**records[index], 'answer': answer})
                                for index, answer in zip(indices, answers)],
            'match_selected_answers': answers,
            'match_answer_extracted': torch.tensor(extracted, dtype=torch.bool),
        }

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        extracted = batch['match_answer_extracted'].float()
        prefix = self._match_metric_prefix + '_match_'
        metrics = {**result.metrics, prefix + 'extracted_fraction': extracted.mean(),
                   prefix + 'gold_fallback_fraction': (1 - extracted).mean() * (self._match_answer_fallback == 'gold')}
        if self._match_answer_fallback == 'empty':
            metrics[prefix + 'empty_fallback_fraction'] = (1 - extracted).mean()
        return ObjectiveResult(result.objective, metrics, result.valid_token_count,
                               result.token_weighted_metrics, result.requires_optimizer_step)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'answer_context': 'single_turn',
                'answer_extraction_version': 1,
                'answer_source': 'own_rollout_with_' + self._match_answer_fallback + '_fallback'}
