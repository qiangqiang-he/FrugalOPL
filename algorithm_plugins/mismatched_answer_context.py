"""Per-rollout answer-conflict context shared by forward and reverse alignment."""
from collections.abc import Mapping, Sequence

import torch

from utils.answer_mismatch import choose_mismatched_answers
from .algorithm_base import ObjectiveResult


class MismatchedAnswerTeacherMixin:
    _mismatch_metric_prefix = 'fkl'

    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        self._answer_selection_seed = runtime.seed

    def collect_actor_signals(self, signals):
        request = super().collect_actor_signals(signals)
        request['return_signals'] = {**request['return_signals'], 'responses': True}
        return request

    def collect_learner_signals(self, signals):
        records = self._training_records(signals)
        trajectory = signals.get('traj')
        if not isinstance(trajectory, Mapping):
            raise ValueError('Answer-conflict alignment requires a trajectory mapping.')
        indices = trajectory.get('prompt_indices')
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError('Answer-conflict prompt_indices must be a one-dimensional integer tensor.')
        indices = indices.tolist()
        responses = trajectory.get('responses', signals.get('responses'))
        if responses is None:
            responses = [''] * len(indices)
        if (isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence)
                or len(responses) != len(indices) or not all(isinstance(value, str) for value in responses)):
            raise ValueError('Answer-conflict responses must be question-aligned strings.')
        batch_id = signals.get('batch_id')
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError('Answer-conflict alignment requires a stable batch_id.')
        choices = choose_mismatched_answers(records, indices, responses, seed=self._answer_selection_seed,
            batch_id=batch_id, expected_replies=int(self._sampling['n']))
        return {
            'teacher_prompts': [self._render_privileged_prompt({**records[index], 'answer': choice['answer']})
                                for index, choice in zip(indices, choices)],
            'mismatch_selected_answers': [choice['answer'] for choice in choices],
            'mismatch_source': torch.tensor([choice['source'] for choice in choices], dtype=torch.long),
            'mismatch_peer_indices': torch.tensor([choice['peer_index'] for choice in choices], dtype=torch.long),
            'mismatch_candidate_count': torch.tensor([choice['candidate_count'] for choice in choices], dtype=torch.float32),
            'mismatch_own_missing': torch.tensor([choice['own_missing'] for choice in choices], dtype=torch.bool),
        }

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        source = batch['mismatch_source']
        prefix = self._mismatch_metric_prefix + '_mismatch_'
        metrics = {**result.metrics,
            prefix + 'peer_fraction': (source == 0).float().mean(),
            prefix + 'perturbed_gold_fraction': (source == 1).float().mean(),
            prefix + 'gold_escape_fraction': (source == 2).float().mean(),
            prefix + 'missing_own_answer_fraction': batch['mismatch_own_missing'].float().mean(),
            prefix + 'candidate_count': batch['mismatch_candidate_count'].float().mean()}
        return ObjectiveResult(result.objective, metrics, result.valid_token_count,
                               result.token_weighted_metrics, result.requires_optimizer_step)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'answer_selection_version': 1,
                'answer_selection_seed': self._answer_selection_seed}
