"""Protect coordinate mapping, teacher/student parity and frozen source data."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import qreflector_geometry_prompt as geometry
import qreflector_sol_round2 as round2
from PIL import Image


def sample(root, action=None):
    Image.new('RGB', (1080, 2400), 'red').save(root / 'before.png')
    Image.new('RGB', (1080, 2400), 'blue').save(root / 'after.png')
    context = {'goal': 'Open expense entry', 'history': [], 'action': action or {
        'type': 'Tap', 'arguments': {'x': 950, 'y': 2000}, 'description': 'Tap add'}}
    text = ('V9 instruction sentinel\n\n=== Context ===\n' + json.dumps(context) +
            '\n=== Image Reference ===\nBEFORE then AFTER\nRequired output sentinel')
    target = {'precondition': 'REFERENCE_PRE_ONLY', 'postcondition': 'REFERENCE_POST_ONLY',
              'reason': 'Precondition check: P. Postcondition check: Q. Task progress: useful.', 'answer': 'Yes'}
    return {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': text},
            {'type': 'image', 'image': 'before.png'}, {'type': 'image', 'image': 'after.png'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(target)}]}]}


class GeometryTests(unittest.TestCase):
    def test_original_coordinates_map_to_relative_location_without_changing_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = sample(root); original = copy.deepcopy(row)
            augmented = geometry.augment_row(row, root)
            info = geometry.geometry(row, root)
            self.assertEqual(row, original)
            self.assertEqual(geometry.context_of(augmented), geometry.context_of(row))
            self.assertEqual(info['images'][0]['input_width'], 230)
            self.assertEqual(info['images'][0]['input_height'], 512)
            self.assertEqual(info['current_action_points'][0]['original_pixels'], [950, 2000])
            self.assertEqual(info['current_action_points'][0]['normalized_xy'], [0.87963, 0.833333])
            self.assertEqual(augmented['messages'][1], original['messages'][1])
            self.assertEqual(augmented['messages'][0]['content'][1:], original['messages'][0]['content'][1:])
            stripped = augmented['messages'][0]['content'][0]['text'].replace(geometry.geometry_block(row, root), '')
            self.assertEqual(stripped, original['messages'][0]['content'][0]['text'])

    def test_raw_and_already_augmented_rows_produce_identical_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = sample(root)
            once = geometry.augment_row(row, root)
            self.assertEqual(once, geometry.augment_row(once, root))
            stale = copy.deepcopy(once)
            stale['messages'][0]['content'][0]['text'] = stale['messages'][0]['content'][0]['text'].replace('0.87963', '0.1')
            with self.assertRaisesRegex(ValueError, 'differs'):
                geometry.augment_row(stale, root)

    def test_teacher_sees_exact_student_prefix_but_reference_is_teacher_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = sample(root)
            student = round2.correction.student_messages(geometry.augment_row(row, root))
            cfg = {'data': str(root), 'image_max_side': 512, 'teacher_model': 'gpt-6.1-sol',
                   'teacher_reasoning': 'medium', 'teacher_max_output_tokens': 4096}
            request = round2.teacher_request(row, 'STUDENT_DRAFT', cfg)
            content = request['input'][0]['content']
            self.assertEqual(content[0]['text'], student[0]['content'][0]['text'])
            self.assertEqual(content[0]['text'].count(geometry.MARKER), 1)
            self.assertNotIn('REFERENCE_PRE_ONLY', json.dumps(student))
            self.assertIn('REFERENCE_PRE_ONLY', content[-1]['text'])
            self.assertEqual([c['type'] for c in content], ['input_text', 'input_image', 'input_image', 'input_text'])

    def test_geometry_does_not_depend_on_reference_or_after_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = sample(root)
            block = geometry.geometry_block(row, root)
            row['messages'][1]['content'][0]['text'] = 'Different target, including a different label'
            Image.new('RGB', (1080, 2400), 'green').save(root / 'after.png')
            self.assertEqual(geometry.geometry_block(row, root), block)

    def test_swipe_both_endpoints_and_out_of_bounds_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = sample(root, {'type': 'Swipe', 'arguments': {
                'x1': 540, 'y1': 1800, 'x2': 1081, 'y2': -20}, 'description': 'Swipe'})
            points = geometry.geometry(row, root)['current_action_points']
            self.assertEqual(points[0]['normalized_xy'], [0.5, 0.75])
            self.assertFalse(points[1]['inside_original_image'])
            self.assertGreater(points[1]['normalized_xy'][0], 1)
            self.assertLess(points[1]['normalized_xy'][1], 0)

    def test_noncoordinate_actions_have_no_invented_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for action in [{'type': 'Back', 'arguments': None},
                           {'type': 'Open_App', 'arguments': {'app_name': 'Markor'}},
                           {'type': 'Unknown', 'arguments': {'raw': 'opaque'}}]:
                self.assertEqual(geometry.geometry(sample(root, action), root)['current_action_points'], [])

    def test_incomplete_and_invalid_coordinate_pairs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for args in [{'x': 20}, {'x': float('nan'), 'y': 30}, {'x': True, 'y': 30}]:
                with self.assertRaises(ValueError):
                    geometry.geometry(sample(root, {'type': 'Tap', 'arguments': args}), root)


if __name__ == '__main__':
    unittest.main()
