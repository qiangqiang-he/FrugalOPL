"""TRA ablations that vary only the frozen Teacher's privileged context."""
from collections.abc import Mapping
import hashlib
import json
import random

import torch

from utils.algorithm_runtime import apply_generation_chat_template, controller_questions_and_answers
from utils.eval import load_dataset_records
from utils.prompts import TRA_TRUE_ANSWER_PROMPT, render_prompt_from_record
from .plugin_registry import register_plugin_class
from .tra_plugin import TRAPlugin


def _question_records(signals):
    questions, _ = controller_questions_and_answers(signals)
    records = signals.get('records')
    if not isinstance(records, list) or len(records) != len(questions):
        raise ValueError('TRA requires question-aligned training records.')
    if not all(isinstance(record, Mapping) and isinstance(record.get('question'), str)
               and record['question'].strip() for record in records):
        raise ValueError('TRA requires a non-empty question in every record.')
    return records


@register_plugin_class
class TRASelfShotPlugin(TRAPlugin):
    """The existing TRA Teacher, explicitly named tra_self_shot."""

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'tra_self_shot'}

    def load_training_state_dict(self, state, *, step):
        # The old TRA entry has exactly this same Teacher context and target.
        if isinstance(state, Mapping) and state.get('algorithm') == 'TRA':
            state = dict(state)
            if self._ratio_power == 1.0 and 'ratio_power' not in state:
                state.update(ratio_power=1.0, target_power=float(state.get('target_power', 2)))
            state['algorithm'] = 'tra_self_shot'
        return super().load_training_state_dict(state, step=step)


@register_plugin_class
class TRARandomShotPlugin(TRAPlugin):
    """Uniform full-training-set demonstration, sampled per question occurrence."""

    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        if runtime.training_dataset is None:
            raise ValueError('tra_random_shot requires controller.data.train.')
        self._dataset_config = dict(runtime.training_dataset)
        self._demo_seed = runtime.seed
        self._demo_pool = None
        self._demo_fingerprint = None

    @staticmethod
    def _training_records(signals):
        return _question_records(signals)

    def _load_demo_pool(self):
        if self._demo_pool is None:
            records = load_dataset_records(self._dataset_config)
            pool = []
            for record in records:
                if not isinstance(record.get('solution'), str) or not record['solution'].strip():
                    raise ValueError('tra_random_shot requires a solution for every demonstration candidate.')
                pool.append({'question': record['question'], 'solution': record['solution']})
            self._demo_pool = pool
            self._demo_fingerprint = hashlib.sha256(json.dumps(pool, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return self._demo_pool

    def _demo_index(self, record, batch_id):
        pool = self._load_demo_pool()
        # Rank, shard offset and scheduling order are intentionally absent.
        identity = json.dumps(['tra_random_shot_v1', self._demo_seed, batch_id, record['question']],
                              ensure_ascii=False, separators=(',', ':'))
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest(), 'big')
        return random.Random(seed).randrange(len(pool))

    def collect_learner_signals(self, signals):
        batch_id = signals.get('batch_id')
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError('tra_random_shot requires a stable batch_id for reproducible sampling.')
        pool = self._load_demo_pool()
        prepared = []
        indices = []
        for record in self._training_records(signals):
            index = self._demo_index(record, batch_id)
            demo = pool[index]
            # The inherited limiter may shorten only this example solution.
            prepared.append({**record, 'solution': demo['solution'],
                             '_tra_demo_question': demo['question']})
            indices.append(index)
        additions = super().collect_learner_signals({**signals, 'records': prepared})
        prompt_indices = signals['traj']['prompt_indices'].tolist()
        additions['tra_demo_indices'] = torch.tensor([indices[index] for index in prompt_indices], dtype=torch.long)
        return additions

    def _render_teacher_record(self, record):
        if '_tra_demo_question' not in record:
            raise ValueError('Random demonstrations must be selected in collect_learner_signals().')
        question = render_prompt_from_record(self._train_template, record)
        example = render_prompt_from_record(self._train_template,
                                             {**record, 'question': record['_tra_demo_question']})
        if self._self_shot_format == 'conversation':
            messages = [{'role': 'user', 'content': example},
                        {'role': 'assistant', 'content': record['solution']},
                        {'role': 'user', 'content': question}]
        else:
            messages = [{'role': 'user', 'content': 'Here is a solved example:\n\n'
                         + example + '\n\nSolution:\n' + record['solution']
                         + '\n\nNow solve this problem:\n\n' + question}]
        return apply_generation_chat_template(self._tokenizer, messages, enable_thinking=self._enable_thinking)

    def training_state_dict(self):
        self._load_demo_pool()
        return {**super().training_state_dict(), 'algorithm': 'tra_random_shot',
                'demo_sampling_version': 1, 'demo_seed': self._demo_seed,
                'demo_pool_fingerprint': self._demo_fingerprint, 'demo_pool_size': len(self._demo_pool)}


@register_plugin_class
class TRATrueAnswerPlugin(TRAPlugin):
    """A single current-question user turn containing only its correct answer."""

    @staticmethod
    def _training_records(signals):
        records = _question_records(signals)
        if not all(record.get('answer') is not None and str(record['answer']).strip() for record in records):
            raise ValueError('tra_true_answer requires a non-empty correct answer in every record.')
        return records

    def _render_teacher_record(self, record):
        if record.get('answer') is None or not str(record['answer']).strip():
            raise ValueError('tra_true_answer requires the current question correct answer.')
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

    def training_state_dict(self):
        state = super().training_state_dict()
        state.pop('self_shot_format', None)
        return {**state, 'algorithm': 'tra_true_answer', 'answer_prompt_version': 1}
