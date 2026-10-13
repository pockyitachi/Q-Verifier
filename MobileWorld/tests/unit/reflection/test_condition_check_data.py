import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import qreflector_condition_check_data as data
import refine_condition_check_annotations as refine
from check_condition_annotation_coordinates import map_coordinates


class ConditionCheckDataTests(unittest.TestCase):
    def setUp(self):
        self.source = {'precondition': 'The search field is visible.',
                       'postcondition': 'The requested search is submitted.',
                       'reason': 'SECRET_OLD_REASON', 'answer': 'Yes'}
        self.annotation = {
            'precondition_check': 'The BEFORE screen shows an editable search field at the top.',
            'precondition_status': 'supported',
            'postcondition_check': 'The AFTER screen shows no requested query or search results.',
            'postcondition_status': 'contradicted',
            'task_progress_check': 'No useful progress toward finding the requested song is shown.',
            'task_progress_status': 'contradicted',
            'evidence_verdict': 'No', 'conditions_suitable': True, 'audit_note': ''}

    def test_preserves_label_and_conditions_even_when_teacher_disagrees(self):
        before = copy.deepcopy(self.source)
        result = data.assemble_target(self.source, self.annotation)
        self.assertEqual(list(result), list(data.FIELDS))
        self.assertEqual(result['answer'], 'Yes')
        self.assertEqual(result['precondition'], before['precondition'])
        self.assertEqual(result['postcondition'], before['postcondition'])
        self.assertEqual(self.source, before)
        self.assertTrue(result['reason'].startswith('Precondition check:'))
        self.assertLess(result['reason'].index('Postcondition check:'), result['reason'].index('Task progress:'))

    def test_rejects_inconsistent_yes_and_placeholder_prose(self):
        with self.assertRaises(ValueError):
            data.validate_annotation({**self.annotation, 'evidence_verdict': 'Yes'})
        with self.assertRaises(ValueError):
            data.validate_annotation({**self.annotation, 'precondition_check': 'Looks good.'})

    def test_teacher_request_does_not_leak_old_verdict_or_reason(self):
        row = {'messages': [{'role': 'user', 'content': []},
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(self.source)}]}]}
        user = {'role': 'user', 'content': [{'type': 'text', 'text': 'TASK_CONTEXT'}]}
        plan = {'model': 'gpt-6-luna', 'reasoning_effort': 'medium',
                'max_output_tokens': 3072, 'image_max_side': 1280}
        with patch.object(data, 'load_images', return_value=[Image.new('RGB', (3, 3), color)
                                                           for color in ('red', 'blue')]):
            request, hashes = data.build_request(row, user, plan)
        text = request['input'][0]['content'][0]['text']
        self.assertNotIn('SECRET_OLD_REASON', text)
        self.assertNotIn('"answer": "Yes"', text)
        self.assertIn(self.source['postcondition'], text)
        self.assertIn('TASK_CONTEXT', text)
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertEqual(request['input'][0]['content'][1]['text'], 'BEFORE')
        self.assertEqual(request['input'][0]['content'][3]['text'], 'AFTER')

    def test_test_split_is_never_sent_to_annotation_api(self):
        client = object.__new__(data.Client)
        with self.assertRaisesRegex(ValueError, 'Test annotation is prohibited'):
            client.generate('test', 1, {}, {})

    def test_original_coordinates_are_scaled_not_treated_as_out_of_bounds(self):
        args = {'x': 815, 'y': 1083, 'text': 'not a coordinate'}
        mapped = map_coordinates(1440, 3120, 1280, args)
        self.assertEqual(mapped['uploaded_image_width_height'], [590, 1280])
        self.assertEqual(mapped['equivalent_coordinates_on_uploaded_image'], {'x': 333.92, 'y': 444.31})
        self.assertEqual(args, {'x': 815, 'y': 1083, 'text': 'not a coordinate'})

    def test_condition_repair_sees_only_before_and_no_original_label_or_reason(self):
        user = {'role': 'user', 'content': [
            {'type': 'text', 'text': 'instructions\n=== Context ===\n{}\n=== Image Reference ===\nrefs'},
            {'type': 'image', 'image': 'before.png'}, {'type': 'image', 'image': 'after.png'}]}
        row = {'messages': [user, {'role': 'assistant', 'content': [
            {'type': 'text', 'text': json.dumps(self.source)}]}]}
        payload = {'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'context'},
            {'type': 'input_text', 'text': 'BEFORE'},
            {'type': 'input_image', 'image_url': 'BEFORE_PIXELS'},
            {'type': 'input_text', 'text': 'AFTER'},
            {'type': 'input_image', 'image_url': 'AFTER_PIXELS'}]}],
            'text': {'format': {'name': 'old', 'schema': {}}}}
        annotation = {**self.annotation, 'conditions_suitable': False}
        with patch.object(data, 'build_request', return_value=(payload, ['before_hash', 'after_hash'])), \
                patch('PIL.Image.open', return_value=Image.new('RGB', (1440, 3120))), \
                patch.object(refine, 'send', side_effect=[
                    {'value': {'precondition': 'Fixed pre.', 'postcondition': 'Fixed post.', 'repair_note': 'Fix.'}},
                    {'sample_id': 'train-1', 'value': self.annotation}]) as send:
            refine.refine_one(Path('/tmp/unused'), {}, 'NEVER_SERIALIZE_KEY', 'train', 1, row, user,
                              {'annotation': annotation})
        first = json.dumps(send.call_args_list[0].args[0])
        self.assertIn('BEFORE_PIXELS', first)
        self.assertNotIn('AFTER_PIXELS', first)
        self.assertNotIn('SECRET_OLD_REASON', first)
        self.assertNotIn('"answer": "Yes"', first)
        self.assertNotIn('NEVER_SERIALIZE_KEY', first)


if __name__ == '__main__':
    unittest.main()
