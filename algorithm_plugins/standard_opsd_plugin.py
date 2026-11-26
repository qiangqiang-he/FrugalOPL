"""Clipped OPSD with the upstream single-turn reference-solution context."""

from dataclasses import replace
from .kl_opsd_plugin import KLOPSDPlugin
from .plugin_registry import register_plugin_class


@register_plugin_class
class StandardOPSDPlugin(KLOPSDPlugin):
    """Use one question, its solution, and the official transition instruction.

    KLOPSDPlugin already renders the upstream reason_first=False Teacher
    template. Reuse its solution-only prompt limiter and clipped top-k
    Forward KL; no solved-question conversation is prepended.
    """

    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('kl_token_clip', 1.0e-7) is None:
            raise ValueError('standard_OPSD requires a positive kl_token_clip.')
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def compute_objective(self, batch):
        result = super().compute_objective(batch)
        return replace(
            result,
            metrics={key.replace('kl_opsd_', 'standard_opsd_', 1): value
                     for key, value in result.metrics.items()},
            token_weighted_metrics=frozenset(key.replace('kl_opsd_', 'standard_opsd_', 1)
                                             for key in result.token_weighted_metrics),
        )

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'standard_OPSD',
                'teacher_context': 'single_turn_reference_solution'}


__all__ = ['StandardOPSDPlugin']
