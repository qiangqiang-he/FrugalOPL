"""Forward KL with each rollout's own answer, falling back to the gold answer."""
from .fkl_true_answer_plugin import FKLTrueAnswerPlugin
from .matched_answer_context import MatchedAnswerTeacherMixin
from .plugin_registry import register_plugin_class


@register_plugin_class
class FKLMatchedAnswerPlugin(MatchedAnswerTeacherMixin, FKLTrueAnswerPlugin):
    _match_metric_prefix = 'fkl'

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'FKL_match_answer'}
