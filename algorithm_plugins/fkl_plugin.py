"""FKL: forward KL toward the frozen privileged Teacher."""
from dataclasses import replace

from .algorithm_base import ObjectiveResult
from .plugin_registry import register_plugin_class
from .self_shot_opsd_plugin import SelfShotOPSDPlugin


@register_plugin_class
class FKLPlugin(SelfShotOPSDPlugin):
    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('kl_token_clip') is not None:
            raise ValueError('FKL requires kl_token_clip: null.')
        # KLOPSDPlugin's historical default is clipped; FKL is never clipped.
        runtime = replace(runtime, algorithm={**runtime.algorithm, 'kl_token_clip': None})
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        metrics = {key.replace('self_shot_opsd_', 'fkl_', 1): value for key, value in result.metrics.items()}
        return ObjectiveResult(result.objective, metrics, result.valid_token_count)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'FKL'}
