"""Teacher reverse align: fit detached p*(p/q)^n on plausible support."""
import math
from collections.abc import Mapping
from utils.tensor_signals import masked_token_mean

from .algorithm_base import ObjectiveResult
from .plugin_registry import register_plugin_class
from .ratio_opsd_plugin import RatioOPSDPlugin


@register_plugin_class
class TRAPlugin(RatioOPSDPlugin):
    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        fraction = runtime.algorithm.get('tra_plausible_fraction', .01)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('tra_plausible_fraction must lie in (0,1].')
        self._plausible_fraction = float(fraction)
        power = runtime.algorithm.get('tra_ratio_power', 1.0)
        if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(power):
            raise ValueError('tra_ratio_power must be a finite real number.')
        self._ratio_power = float(power)

    def compute_objective(self, batch):
        values = self.compute_ratio_kl(
            batch, teacher_prompts=batch.get('teacher_prompts'),
            temperature=self._kl_temperature, plausible_fraction=self._plausible_fraction,
            ratio_power=self._ratio_power,
        )
        mask = batch['response_mask'].bool()
        loss = masked_token_mean(values['ratio_kl_loss'], mask)
        metrics = {name: masked_token_mean(value, mask).detach() for name, value in values.items()}
        metrics['tra_loss'] = metrics.pop('ratio_kl_loss')
        metrics['tra_ratio_power'] = loss.detach().new_tensor(self._ratio_power)
        return ObjectiveResult(-loss, metrics, mask.sum().detach(), frozenset(metrics))

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'TRA',
                'ratio_power': self._ratio_power, 'target_power': 1 + self._ratio_power}

    def load_training_state_dict(self, state, *, step):
        # Existing TRA checkpoints used fixed n=1 and had no ratio_power field.
        if isinstance(state, Mapping) and self._ratio_power == 1.0 and state.get('algorithm') == 'TRA' and 'ratio_power' not in state:
            state = {**state, 'ratio_power': 1.0, 'target_power': float(state.get('target_power', 2))}
        return super().load_training_state_dict(state, step=step)
