"""Unclipped Forward KL with a different answer selected for each rollout."""
from .fkl_true_answer_plugin import FKLTrueAnswerPlugin
from .mismatched_answer_context import MismatchedAnswerTeacherMixin
from .plugin_registry import register_plugin_class


@register_plugin_class
class FKLMismatchedAnswerPlugin(MismatchedAnswerTeacherMixin, FKLTrueAnswerPlugin):
    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'FKL_mismatched_answer'}
