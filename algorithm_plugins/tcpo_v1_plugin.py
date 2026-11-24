"""TCPO_v1: Teacher KL on success, TCA on error/cut, question-group weights."""
import torch
from utils.tcdo import CORRECT, INCORRECT, TRUNCATED
from utils.tcpo_groups import trajectory_group_fractions
from utils.tensor_signals import masked_token_mean
from .algorithm_base import ObjectiveResult, Signals
from .plugin_registry import register_plugin_class
from .tcdo_plugin import TCDOPlugin


@register_plugin_class
class TCPOV1Plugin(TCDOPlugin):
    """Compute group proportions before the Learner shards individual rollouts."""

    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        if runtime.algorithm.get('tcpo_objective_version') != 2:
            raise ValueError('TCPO_v1 requires tcpo_objective_version: 2; use TCDOPlugin for the legacy objective.')
        self.correct_weight = self._parameter(runtime, 'tcpo_alpha_plus', .25)
        self.incorrect_weight = self._parameter(runtime, 'tcpo_alpha_minus', 1.)
        self.truncated_weight = self._parameter(runtime, 'tcpo_alpha_tru', 1.)
        self.fraction = self._parameter(runtime, 'tcpo_plausible_fraction', .01, positive=True)
        if self.fraction > 1:
            raise ValueError('tcpo_plausible_fraction must lie in (0,1].')

    def collect_learner_signals(self, signals: Signals) -> Signals:
        additions = super().collect_learner_signals(signals)
        state = additions['tcdo_state']
        fractions = trajectory_group_fractions(state, signals['prompt_indices'], expected_count=self._sampling['n'])
        alpha = torch.tensor([self.correct_weight, self.incorrect_weight, self.truncated_weight], dtype=torch.float64)
        additions['tcpo_group_fractions'] = fractions
        additions['tcpo_sample_weight'] = alpha[state] * fractions.gather(1, state[:, None]).squeeze(1)
        additions['tcdo_batch_stats'].update(alpha_plus=self.correct_weight,
            alpha_minus=self.incorrect_weight, alpha_tru=self.truncated_weight)
        return additions

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        state = int(batch['tcdo_state'].item())
        weight = float(batch['tcpo_sample_weight'].item())
        mask = batch['response_mask'].bool()
        if weight == 0:
            values = self.compute_logits(batch, mode='policy_no_grad', return_signals={
                'response_entropy': True, 'entropy_requires_grad': False})
            entropy = masked_token_mean(values['response_entropy'], mask).detach()
            loss = entropy.new_zeros((), requires_grad=True)
            metrics = {'tcpo_loss': loss.detach(), 'student_entropy_nats': entropy, 'target_entropy_nats': entropy}
        else:
            # Legacy TRUNCATED selects q; legacy INCORRECT selects p²/q.
            # These select target scorers, not the actual trajectory label.
            mode = TRUNCATED if state == CORRECT else INCORRECT
            values = self.compute_tcdo(batch, teacher_prompts=batch['teacher_prompts'], state=mode,
                temperature=self._kl_temperature, plausible_fraction=self.fraction)
            loss = masked_token_mean(values['tcdo_kl_loss'], mask) * weight
            metrics = {name: masked_token_mean(value, mask).detach()
                       for name, value in values.items() if name != 'tcdo_kl_loss'}
            metrics['tcpo_loss'] = loss.detach()
        token_metrics = frozenset(metrics)
        metrics.update(batch['tcdo_batch_stats'])
        rates = batch['tcpo_group_fractions'].reshape(-1)
        metrics.update(group_correct_rate=rates[CORRECT], group_incorrect_rate=rates[INCORRECT],
                       group_truncated_rate=rates[TRUNCATED], trajectory_weight=weight)
        return ObjectiveResult(-loss, metrics, mask.sum().detach(), token_metrics, weight > 0)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'TCPO_v1', 'objective_version': 2,
                'correct_weight': self.correct_weight, 'group_relative_weights': True,
                'group_rollouts_per_prompt': self._sampling['n']}
