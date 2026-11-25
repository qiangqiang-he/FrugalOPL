"""Reverse alignment with the same rollout-mismatched answer selection as FKL."""
from .mismatched_answer_context import MismatchedAnswerTeacherMixin
from .plugin_registry import register_plugin_class
from .tra_privilege_plugins import TRATrueAnswerPlugin


@register_plugin_class
class TRAMismatchedAnswerPlugin(MismatchedAnswerTeacherMixin, TRATrueAnswerPlugin):
    _mismatch_metric_prefix = 'tra'

    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('answer_context', 'single_turn') != 'single_turn':
            raise ValueError('Answer privileges require answer_context: single_turn.')
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'tra_mismatch_answer',
                'answer_context': 'single_turn'}
