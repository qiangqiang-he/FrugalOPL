"""Deterministic, question-local answer conflicts for FKL Teacher prompts."""
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import re

from utils.answer_verifier import answers_equivalent, extract_choice_label, extract_final_answer


def _plain(value):
    text = str(value).strip()
    previous = None
    while previous != text:
        previous, text = text, re.sub(r'\\(?:text|mathrm|textrm|textnormal)\s*\{([^{}]*)\}', r'\1', text)
    return text.replace('\\quad', ' ').replace('\\,', ' ').replace('\\;', ' ').replace('\\ ', ' ').strip()


@lru_cache(maxsize=2048)
def question_options(question):
    pattern = r'(?m)^\s*(?:\(([A-J])\)|([A-J])[.)：:])\s*(.+?)(?=\n\s*(?:\([A-J]\)|[A-J][.)：:])|\Z)'
    return tuple(((match.group(1) or match.group(2)), match.group(3).strip())
                 for match in re.finditer(pattern, question, re.S))


def canonical_answer(question, answer):
    text = _plain(answer)
    options = dict(question_options(question))
    label = extract_choice_label(text)
    if options and label in options:
        text = _plain(options[label])
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]|\$', '', text).strip().rstrip('.')
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    # Common answer units do not distinguish an otherwise identical value.
    if re.search(r'\d', text):
        text = re.sub(r'(?i)\s*(?:hours?|minutes?|seconds?|meters?|metres?|kilometers?|centimeters?|'
                      r'girls?|boys?|people|students?|dollars?|cm|mm|km|m)\s*(?:\^\s*\{?[23]\}?)?\s*$', '', text)
    text = re.sub(r'^([+]?[0-9]+)\s*(?:and\s*)?\\frac\s*\{([0-9]+)\}\s*\{([0-9]+)\}',
                  lambda m: f'({m.group(1)}+({m.group(2)}/{m.group(3)}))', text)
    intervals = re.findall(r'[\[(]\s*[-+\w.\\{}]+\s*,\s*[-+\w.\\{}]+\s*[\])]', text)
    if (len(intervals) == 1 and re.search(r'(?i)range|interval|contained', text)
            and not re.search(r'(?i)\bnot\b|outside', text)):
        text = intervals[0]
    if re.search(r'(?i)\bprove\b|\bshow that\b', question):
        bounds = re.findall(r'\\(?:geq|leq)\s*([-+]?\d+(?:\.\d+)?)', question)
        if bounds:
            compact = lambda value: re.sub(r'\s+|\$|\\[()[\]]', '', _plain(value)).replace(r'\left', '').replace(r'\right', '')
            if text == bounds[-1] or (re.search(r'\\(?:geq|leq)', text) and compact(text) in compact(question)):
                text = 'proof_goal_bound_' + bounds[-1]
    return text.strip()


@lru_cache(maxsize=8192)
def _equivalent_cached(question, first, second):
    a, b = canonical_answer(question, first), canonical_answer(question, second)
    compact = lambda value: re.sub(r'\s+', '', value).replace(r'\left', '').replace(r'\right', '').casefold()
    if compact(a) == compact(b):
        return True
    # The generic numeric verifier may strip an equation/inequality's LHS.
    # Do not compare different relational statements as bare RHS constants.
    relation = r'[<>=]|\\(?:geq|leq|neq|in)\b'
    if re.search(relation, a) or re.search(relation, b):
        return False
    return answers_equivalent(a, b)


def answers_match(question, first, second):
    if first is None or second is None:
        return False
    a, b = sorted((str(first), str(second)))
    return _equivalent_cached(str(question), a, b)


def perturb_gold_answer(question, gold, own_answer):
    """Prefer a changed gold answer that differs from both gold and rollout."""
    gold = str(gold).strip()
    if not gold:
        raise ValueError('Answer-conflict fallback requires a non-empty gold answer.')
    options = question_options(question)
    if len(options) >= 2:
        for label, _ in options:
            if not answers_match(question, label, gold) and not answers_match(question, label, own_answer):
                return label, 1
        if own_answer is not None and not answers_match(question, gold, own_answer):
            return gold, 2
        raise ValueError('Question options contain no mismatched answer.')
    boolean = _plain(gold).casefold()
    opposites = {'yes': 'No', 'no': 'Yes', 'true': 'False', 'false': 'True'}
    if boolean in opposites:
        candidate = opposites[boolean]
        if not answers_match(question, candidate, own_answer):
            return candidate, 1
        return gold, 2
    candidates = []
    number = re.search(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?', gold)
    if number is not None:
        try:
            value = Decimal(number.group())
            for offset in (1, 2, 3, -1, 5):
                replacement = format(value + offset, 'f')
                candidates.append(gold[:number.start()] + replacement + gold[number.end():])
        except InvalidOperation:
            pass
    for multiplier in (-1, 2, 3, 4):
        candidates.append(rf'{multiplier}\cdot\left({gold}\right)')
    for candidate in candidates:
        if not answers_match(question, candidate, gold) and not answers_match(question, candidate, own_answer):
            return candidate, 1
    # Binary answer spaces can have no third distinct answer. The gold answer
    # still provides the requested conflict when the rollout chose its opposite.
    if own_answer is not None and not answers_match(question, gold, own_answer):
        return gold, 2
    raise ValueError('Could not construct an answer different from this rollout.')


def choose_mismatched_answers(records, indices, responses, *, seed, batch_id, expected_replies):
    """Select from complete same-question groups before rollout-level sharding."""
    if len(indices) != len(responses) or not indices:
        raise ValueError('Answer-conflict responses must align with rollout prompt indices.')
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records) for index in indices):
        raise ValueError('Answer-conflict prompt index is outside the question batch.')
    groups = {index: [] for index in range(len(records))}
    extracted = [extract_final_answer(response)[0] or None for response in responses]
    for row, index in enumerate(indices):
        groups[index].append(row)
    if any(len(rows) != expected_replies for rows in groups.values()):
        raise ValueError('Answer-conflict selection requires each complete question rollout group.')
    selected = []
    for row, (index, own) in enumerate(zip(indices, extracted)):
        record = records[index]
        question = record['question']
        eligible = [peer for peer in groups[index] if peer != row and own is not None
                    and extracted[peer] is not None and not answers_match(question, own, extracted[peer])]
        eligible.sort(key=lambda peer: (responses[peer], extracted[peer]))
        if eligible:
            identity = json.dumps([seed, batch_id, question, responses[row]], ensure_ascii=False)
            pick = int.from_bytes(hashlib.sha256(identity.encode()).digest(), 'big') % len(eligible)
            peer = eligible[pick]
            answer, source = extracted[peer], 0
        else:
            peer = -1
            answer, source = perturb_gold_answer(question, record['answer'], own)
        if own is not None and answers_match(question, answer, own):
            raise RuntimeError('Answer-conflict selection produced a matching answer.')
        selected.append({'answer': answer, 'source': source, 'peer_index': peer,
                         'candidate_count': len(eligible), 'own_missing': own is None})
    return selected
