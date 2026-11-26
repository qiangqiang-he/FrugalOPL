"""Unclipped Forward KL with the standard OPSD reference-solution context."""

from dataclasses import replace

from .kl_opsd_plugin import KLOPSDPlugin
from .plugin_registry import register_plugin_class


@register_plugin_class
class StandardFKLPlugin(KLOPSDPlugin):
    """Keep standard_OPSD's single-turn Teacher prompt and omit KL clipping."""

    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('kl_token_clip') is not None:
            raise ValueError('standard_FKL requires kl_token_clip: null.')
        # KLOPSDPlugin's historical default is clipped. An omitted option
        # must also select ordinary, unclipped Forward KL for this plugin.
        runtime = replace(runtime, algorithm={**runtime.algorithm, 'kl_token_clip': None})
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        metrics = {key.replace('kl_opsd_', 'standard_fkl_', 1): value
                   for key, value in result.metrics.items()}
        return replace(
            result,
            metrics=metrics,
            # All values are means over the same valid response tokens.
            # Keep KL diagnostics and entropy on the loss's token weighting.
            token_weighted_metrics=frozenset(metrics),
        )

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'standard_FKL',
                'teacher_context': 'single_turn_reference_solution'}


__all__ = ['StandardFKLPlugin']
