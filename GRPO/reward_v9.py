"""V9 reward: legacy asymmetric correctness plus an explicit syntax bonus.

Use with a dataset column named ``expected_answer`` (TRL forwards extra columns).
No label lookup by prompt text, no keyword/length heuristic, no semantic-quality
claim from the presence or order of condition fields. Does not start training.
The symmetric profile remains available for an ablation. For the asymmetric
profile, the future trainer must preserve class weights (see REWARD_V9.txt).
"""
from __future__ import annotations

import json

FIELDS = ('precondition', 'postcondition', 'reason', 'answer')
CHECKS = ('Precondition check:', 'Postcondition check:', 'Task progress:')
JUDGEMENT_WEIGHTS = {
    'asymmetric': {'Yes': 2.5, 'No': 2.0},
    'symmetric': {'Yes': 1.0, 'No': 1.0},
}


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1:
        message = completion[0]
        if message.get('role') != 'assistant':
            raise ValueError('Expected one assistant completion')
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list) and all(c.get('type') == 'text' for c in content):
            return ''.join(c['text'] for c in content)
    raise ValueError('Unsupported completion representation')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def inspect_completion(completion):
    try:
        obj = json.loads(completion_text(completion), object_pairs_hook=unique_object)
    except (ValueError, TypeError, KeyError, AttributeError):
        return {'answer': None, 'ordered_fields': False, 'reason_sections': False}
    if not isinstance(obj, dict):
        return {'answer': None, 'ordered_fields': False, 'reason_sections': False}
    answer = obj.get('answer')
    answer = answer if isinstance(answer, str) and answer in ('Yes', 'No') else None
    ordered = (tuple(obj) == FIELDS and answer is not None and
               all(isinstance(v, str) and v.strip() for v in obj.values()))
    reason = obj.get('reason', '')
    sections = False
    if isinstance(reason, str) and all(reason.count(c) == 1 for c in CHECKS):
        positions = [reason.index(c) for c in CHECKS]
        if positions == sorted(positions):
            ends = positions[1:] + [len(reason)]
            sections = all(reason[start + len(label):end].strip()
                           for start, end, label in zip(positions, ends, CHECKS))
    return {'answer': answer, 'ordered_fields': ordered, 'reason_sections': bool(sections)}


def score(completion, expected_answer, format_weight=0.1, mode='asymmetric'):
    if expected_answer not in ('Yes', 'No'):
        raise ValueError('Missing/invalid reviewed expected_answer; refusing an unlabelled reward')
    if not isinstance(format_weight, (float, int)) or not 0 <= format_weight <= 0.25:
        raise ValueError('Keep the auxiliary format weight between 0 and 0.25')
    if mode not in JUDGEMENT_WEIGHTS:
        raise ValueError('Unknown judgement reward mode')
    details = inspect_completion(completion)
    correct = details['answer'] == expected_answer
    weights = JUDGEMENT_WEIGHTS[mode]
    if details['answer'] is None:
        # The old -1 invalid penalty was better than an incorrect valid answer
        # (-2 or -2.5). Do not incentivize avoiding a verdict on difficult rows.
        judgement = -max(weights.values())
    else:
        judgement = weights[expected_answer] * (1.0 if correct else -1.0)
    # 0.1 is a configurable format regularizer, not an empirically optimal weight.
    syntax = format_weight if details['ordered_fields'] and details['reason_sections'] else 0.0
    return {**details, 'correct': correct, 'mode': mode, 'judgement_reward': judgement,
            'format_reward': syntax, 'reward': judgement + syntax}


class V9RewardFunction:
    __name__ = 'V9RewardFunction'

    def __init__(self, format_weight=0.1, mode='asymmetric'):
        score('{"answer":"Yes"}', 'Yes', format_weight, mode)
        self.format_weight = format_weight
        self.mode = mode

    def __call__(self, completions, expected_answer=None, **kwargs):
        if not isinstance(expected_answer, (list, tuple)) or len(expected_answer) != len(completions):
            raise ValueError('expected_answer must align one-to-one with completions')
        return [score(completion, answer, self.format_weight, self.mode)['reward']
                for completion, answer in zip(completions, expected_answer)]
