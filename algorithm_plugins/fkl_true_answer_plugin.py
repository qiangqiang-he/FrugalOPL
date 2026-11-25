"""Forward KL toward a Teacher given the correct answer in one user turn."""
from .fkl_plugin import FKLPlugin
from .plugin_registry import register_plugin_class
from .true_answer_context import TrueAnswerTeacherMixin


@register_plugin_class
class FKLTrueAnswerPlugin(TrueAnswerTeacherMixin, FKLPlugin):
    def __init__(self, *, tokenizer, runtime):
        if runtime.algorithm.get('answer_context', 'single_turn') != 'single_turn':
            raise ValueError('Answer privileges require answer_context: single_turn.')
        super().__init__(tokenizer=tokenizer, runtime=runtime)

    def training_state_dict(self):
        state = super().training_state_dict()
        state.pop('self_shot_format', None)
        return {**state, 'algorithm': 'FKL_true_answer', 'answer_prompt_version': 2,
                'answer_context': 'single_turn'}
