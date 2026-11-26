"""TRA with a solved one-shot sampled from a different training question."""
import hashlib
import json
import random

from .plugin_registry import register_plugin_class
from .tra_privilege_plugins import TRARandomShotPlugin


@register_plugin_class
class TRAMismatchedSolutionPlugin(TRARandomShotPlugin):
    """Keep each demo's question/solution paired, excluding the current question."""

    def __init__(self, *, tokenizer, runtime):
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        self._demo_question_indices = None

    def _demo_index(self, record, batch_id):
        pool = self._load_demo_pool()
        if self._demo_question_indices is None:
            indices = {}
            for index, demo in enumerate(pool):
                indices.setdefault(demo['question'].strip(), []).append(index)
            self._demo_question_indices = indices
        question = record['question'].strip()
        excluded = self._demo_question_indices.get(question, ())
        available = len(pool) - len(excluded)
        if available < 1:
            raise ValueError('Mismatched-solution TRA requires a different question in the training set.')
        # Sample uniformly over eligible rows without rejection or rank-local RNG.
        identity = json.dumps(['tra_mismatch_solution_v1', self._demo_seed, batch_id, question],
                              ensure_ascii=False, separators=(',', ':'))
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest(), 'big')
        index = random.Random(seed).randrange(available)
        for blocked in excluded:
            if blocked > index:
                break
            index += 1
        return index

    def training_state_dict(self):
        return {**super().training_state_dict(), 'algorithm': 'tra_mismatch_solution',
                'demo_exclude_same_question': True, 'demo_selection_version': 1}
