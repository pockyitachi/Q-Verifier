import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from eval_gui_reflection_native_androidworld import (
    PROMPT_SOURCE, metrics, native_input, official_template, verify_saved,
)

requires_native_source = pytest.mark.skipif(
    not PROMPT_SOURCE.is_file(),
    reason="Provision the upstream GUI-Reflection source under artifacts/vendor first",
)

def user_record():
    context = {
        'goal': 'GOAL_MUST_NOT_BE_A_SEPARATE_INPUT',
        'history': ['HISTORY_MUST_NOT_LEAK'],
        'action': {'type': 'Long_Press', 'arguments': {'x': 10, 'y': 20},
                   'description': 'Select the note to enable moving it.'},
        'precondition': 'REFERENCE_CONDITION_MUST_NOT_LEAK',
    }
    return {'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '=== Context ===\n' + json.dumps(context) + '\n=== Image Reference ==='},
        {'type': 'image', 'image': 'before.png'}, {'type': 'image', 'image': 'after.png'},
    ]}]}


@requires_native_source
def test_native_prompt_uses_existing_description_and_excludes_extra_context():
    template = official_template(PROMPT_SOURCE)
    sample = native_input(user_record(), template)
    assert 'Select the note to enable moving it.' in sample['prompt']
    assert 'Directly answer Yes or No.' in sample['prompt']
    assert sample['prompt'].count('<image>') == 2
    assert all(marker not in sample['prompt'] for marker in ['MUST_NOT', 'precondition', 'Long_Press'])
    assert sample['images'] == ['before.png', 'after.png']


@requires_native_source
def test_changing_after_cannot_change_purpose_or_prompt():
    before = user_record()
    changed = copy.deepcopy(before)
    changed['messages'][0]['content'][-1]['image'] = 'different_after.png'
    template = official_template(PROMPT_SOURCE)
    a, b = native_input(before, template), native_input(changed, template)
    assert a['purpose'] == b['purpose'] and a['prompt_sha256'] == b['prompt_sha256']
    assert a['images'] != b['images']


@requires_native_source
def test_reference_assistant_message_is_rejected():
    record = user_record()
    record['messages'].append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'Yes'}]})
    with pytest.raises(AssertionError):
        native_input(record, official_template(PROMPT_SOURCE))


@requires_native_source
def test_no_fallback_to_task_goal_when_description_missing():
    record = user_record()
    text = record['messages'][0]['content'][0]['text']
    record['messages'][0]['content'][0]['text'] = text.replace('Select the note to enable moving it.', '')
    with pytest.raises(ValueError, match='Missing action description'):
        native_input(record, official_template(PROMPT_SOURCE))


@requires_native_source
def test_invalid_output_is_counted_wrong_and_resume_rejects_stale_prompt():
    prepared = [dict(sample_id=1, **native_input(user_record(), official_template(PROMPT_SOURCE)))]
    labels = [{'sample_id': 'test-1', 'answer': 'No'}]
    row = {'sample_id': 1, 'source_sample_id': 'test-1', 'expected': 'no',
           'predicted': None, 'correct': False, 'full_response': 'Uncertain',
           'official_correct': False, 'prompt_sha256': prepared[0]['prompt_sha256']}
    verify_saved([row], prepared, labels)
    result = metrics([row])
    assert result['total'] == 1 and result['accuracy'] == 0 and result['invalid_outputs'] == 1
    row['prompt_sha256'] = 'different-prompt'
    with pytest.raises(AssertionError):
        verify_saved([row], prepared, labels)
