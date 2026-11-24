"""Teacher reverse align: fit detached p²/q on Student's plausible support."""
import math

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

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        metrics = dict(result.metrics)
        metrics['tra_loss'] = metrics.pop('ratio_kl_loss')
        return ObjectiveResult(result.objective, metrics, result.valid_token_count,
                               frozenset(metrics), result.requires_optimizer_step)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'TRA'}
