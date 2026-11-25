"""Reverse alignment with each rollout's own final answer as Teacher privilege."""
from .matched_answer_context import MatchedAnswerTeacherMixin
from .plugin_registry import register_plugin_class
from .tra_privilege_plugins import TRATrueAnswerPlugin


@register_plugin_class
class TRAMatchedAnswerPlugin(MatchedAnswerTeacherMixin, TRATrueAnswerPlugin):
    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'tra_match_answer'}
