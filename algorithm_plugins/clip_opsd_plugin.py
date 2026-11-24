"""Clip-OPSD: elementwise clipped forward KL on privileged Teacher targets."""
from .algorithm_base import ObjectiveResult
from .plugin_registry import register_plugin_class
from .self_shot_opsd_plugin import SelfShotOPSDPlugin


@register_plugin_class
class ClipOPSDPlugin(SelfShotOPSDPlugin):
    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('kl_token_clip', 1.0e-7) is None:
            raise ValueError('Clip-OPSD requires a positive kl_token_clip.')
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        metrics = {key.replace('self_shot_opsd_', 'clip_opsd_', 1): value for key, value in result.metrics.items()}
        return ObjectiveResult(result.objective, metrics, result.valid_token_count)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'Clip-OPSD'}
