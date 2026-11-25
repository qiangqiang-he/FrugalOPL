"""Forward KL toward a Teacher given a same-question, answer-only one-shot."""
from .fkl_plugin import FKLPlugin
from .plugin_registry import register_plugin_class
from .true_answer_context import TrueAnswerTeacherMixin


@register_plugin_class
class FKLTrueAnswerPlugin(TrueAnswerTeacherMixin, FKLPlugin):
    def _render_teacher_record(self, record):
        if record.get('answer') is None or not str(record['answer']).strip():
            raise ValueError('FKL_true_answer requires the current question correct answer.')
        # FKL and TRA inherit this exact same-question one-shot renderer.
        # Supply the known answer as the assistant demonstration payload.
        return FKLPlugin._render_teacher_record(self, {**record, 'solution': str(record['answer'])})

    def training_state_dict(self):
        state = super().training_state_dict()
        return {**state, 'algorithm': 'FKL_true_answer', 'answer_prompt_version': 1,
                'answer_context': 'one_shot_conversation'}
