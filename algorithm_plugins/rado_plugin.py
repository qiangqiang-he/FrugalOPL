"""Reward-aligned, bidirectional Student-top-k distribution optimization."""
from __future__ import annotations

import math
import torch

from utils.answer_verifier import verify_response_answers
from utils.rado import direction_estimate
from utils.tensor_signals import masked_token_mean
from .algorithm_base import ObjectiveResult
from .plugin_registry import register_plugin_class
from .self_shot_opsd_plugin import SelfShotOPSDPlugin


@register_plugin_class
class RADOPlugin(SelfShotOPSDPlugin):
    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        cfg = runtime.algorithm
        self.delta = float(cfg.get('rado_eta_step', .2))
        self.max_updates = cfg.get('rado_max_updates_per_batch', 4)
        self.ess_fraction = float(cfg.get('rado_min_ess_fraction', .5))
        self.confidence = float(cfg.get('rado_confidence', .95))
        self.bootstrap_samples = cfg.get('rado_bootstrap_samples', 4096)
        if not math.isfinite(self.delta) or not 0 < self.delta <= 1:
            raise ValueError('rado_eta_step must lie in (0,1].')
        if type(self.max_updates) is not int or not 1 <= self.max_updates <= 4:
            raise ValueError('RADO supports 1 to 4 updates per rollout batch.')
        if not 0 < self.ess_fraction <= 1 or not 0 < self.confidence < 1:
            raise ValueError('Invalid RADO confidence or ESS fraction.')
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 100:
            raise ValueError('RADO bootstrap_samples must be >=100.')
        if (self._sampling['temperature'] != 1 or self._sampling['top_p'] != 1
                or self._sampling.get('top_k', -1) not in (-1, 0)):
            raise ValueError('RADO requires unmodified full-support sampling T=1, top_p=1, top_k=-1.')
        if self._kl_temperature != 1 or self._kl_token_clip is not None or cfg.get('teacher_sync_interval', -1) != -1:
            raise ValueError('RADO requires T=1, no CLIP, and a frozen base Teacher.')
        self._window = None

    def collect_actor_signals(self, signals):
        request = super().collect_actor_signals(signals)
        request['return_signals'] = {**request['return_signals'], 'responses': True,
                                    'response_truncated': True, 'response_logprobs': True}
        return request

    def collect_learner_signals(self, signals):
        additions = super().collect_learner_signals(signals)
        batch_id = signals['batch_id']
        fresh = self._window is None or self._window['batch_id'] != batch_id
        if fresh:
            indices = signals['prompt_indices'].tolist()
            cut = signals['response_truncated'].bool()
            rewards = torch.zeros(len(indices), dtype=torch.float64)
            active = (~cut).nonzero().flatten().tolist()
            values = verify_response_answers([signals['responses'][i] for i in active],
                [str(signals['answers'][indices[i]]) for i in active])
            rewards[active] = torch.tensor(values, dtype=torch.float64)
            behavior = signals.get('response_logprobs')
            if behavior is None or len(behavior) != len(indices):
                raise ValueError('RADO needs actual Actor behavior log probabilities, including EOS.')
            lengths = [len(x) for x in signals['response_ids']]
            if any(len(lp) != n for lp, n in zip(behavior, lengths, strict=True)):
                raise ValueError('RADO behavior log probabilities must include all generated tokens.')
            log_mu = torch.tensor([float(torch.as_tensor(lp).double().sum()) for lp in behavior], dtype=torch.float64)
            self._window = {'batch_id': batch_id, 'rewards': rewards, 'log_mu': log_mu,
                            'initial_direction': 0, 'reuse_index': 0}
        window = self._window
        current = self.collect_rado_scores(signals, teacher_prompts=additions['teacher_prompts'])
        log_weights = current['student_sampled_logp'] - window['log_mu']
        decision = direction_estimate(current['rado_token_score'], window['rewards'],
            signals['prompt_indices'], log_weights, delta=self.delta, ess_fraction=self.ess_fraction,
            confidence=self.confidence, bootstrap_samples=self.bootstrap_samples,
            expected_group_size=self._sampling['n'],
            seed=42 + int(batch_id.rsplit('-', 1)[-1])*7 + window['reuse_index'])
        direction = decision['direction']
        if direction and window['initial_direction'] and direction != window['initial_direction']:
            decision.update(eta=0., direction=0, reason='reliable_direction_reversal')
        elif direction and not window['initial_direction']:
            window['initial_direction'] = direction
        tokens = sum(len(x) for x in signals['response_ids'])
        previous_reward = window.get('previous_reward_is')
        decision.update(batch_index=int(batch_id.rsplit('-', 1)[-1]),
                        reuse_index=window['reuse_index'],
                        truncated_fraction=float(signals['response_truncated'].double().mean()),
                        generated_length_mean=sum(len(x) for x in signals['response_ids'])/len(signals['response_ids']),
                        eta_positive=int(decision['eta'] > 0), eta_negative=int(decision['eta'] < 0),
                        eta_zero=int(decision['eta'] == 0),
                        ess_min_fraction=self.ess_fraction,
                        scored_student_entropy_nats=float(current['student_entropy_nats'].sum())/tokens,
                        scored_student_teacher_kl_pq=float(current['student_teacher_kl_pq'].sum())/tokens,
                        reward_is_change=0. if previous_reward is None else decision['reward_is']-previous_reward,
                        has_previous_update=int(previous_reward is not None),
                        behavior_logp_gap_mean=float(log_weights.mean()))
        window['previous_reward_is'] = decision['reward_is']
        window['reuse_index'] += 1
        # Sequence-level audit data are small and retained for every decision.
        return {**additions, 'rado_decision': decision,
                'rado_audit': {key: value.tolist() for key, value in current.items()},
                'rado_rewards': window['rewards'], 'rado_log_weights': log_weights}

    def compute_objective(self, batch):
        decision = batch['rado_decision']
        if decision['eta'] == 0:
            raise RuntimeError('An uncertain RADO batch must not enter the optimizer.')
        values = self.compute_rado(batch, teacher_prompts=batch['teacher_prompts'], eta=decision['eta'])
        mask = batch['response_mask'].bool()
        loss = masked_token_mean(values['rado_kl_loss'], mask)
        metrics = {key: masked_token_mean(value, mask).detach()
                   for key, value in values.items() if key not in ('rado_token_score', 'student_sampled_logp', 'teacher_sampled_logp')}
        token_metrics = frozenset(metrics)
        metrics.update({key: value for key, value in decision.items() if isinstance(value, (int, float))})
        return ObjectiveResult(-loss, metrics, mask.sum().detach(), token_metrics)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'RADO', 'support': 'student_top_k',
                'action_logprob_support': 'full_sampling_distribution',
                'direction_support': 'student_top_k_preserve_mass',
                'score_kl': 'p||q', 'score_reduction': 'sequence_sum', 'baseline': 'group_loo_raw_reward',
                'eta_step': self.delta, 'max_updates': self.max_updates,
                'min_ess_fraction': self.ess_fraction, 'confidence': self.confidence,
                'bootstrap_samples': self.bootstrap_samples}
