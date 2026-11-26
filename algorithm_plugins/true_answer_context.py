"""Answer-only privileged Teacher context shared by forward and reverse alignment."""
from collections.abc import Mapping

from utils.algorithm_runtime import apply_generation_chat_template, controller_questions_and_answers
from utils.prompts import TRA_TRUE_ANSWER_PROMPT, render_prompt_from_record


class TrueAnswerTeacherMixin:
    """Keep the known answer in the Teacher prompt and out of Student prompts."""

    @staticmethod
    def _training_records(signals):
        questions, _ = controller_questions_and_answers(signals)
        records = signals.get('records')
        if not isinstance(records, list) or len(records) != len(questions):
            raise ValueError('True-answer Teacher requires question-aligned training records.')
        if not all(isinstance(record, Mapping) and isinstance(record.get('question'), str)
                   and record['question'].strip() for record in records):
            raise ValueError('True-answer Teacher requires a non-empty question in every record.')
        if not all(record.get('answer') is not None and str(record['answer']).strip() for record in records):
            raise ValueError('True-answer Teacher requires a non-empty correct answer in every record.')
        return records

    def _render_teacher_record(self, record):
        return self._render_answer_teacher_record(record)

    def _render_answer_teacher_record(self, record, *, allow_empty=False):
        if record.get('answer') is None or (not allow_empty and not str(record['answer']).strip()):
            raise ValueError('True-answer Teacher requires the current question correct answer.')
        prompt = render_prompt_from_record(TRA_TRUE_ANSWER_PROMPT,
            {'student_prompt': render_prompt_from_record(self._train_template, record),
             'answer': str(record['answer'])})
        return apply_generation_chat_template(self._tokenizer, [{'role': 'user', 'content': prompt}],
                                              enable_thinking=self._enable_thinking)

    def _render_privileged_prompt(self, record):
        prompt = self._render_teacher_record(record)
        # Never silently shorten the current question or its known answer.
        self._assert_prompt_within_limit(prompt, kind='true-answer Teacher')
        return prompt
