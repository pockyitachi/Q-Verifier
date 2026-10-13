"""Protections for a paired correction pilot, without GPU or network access."""
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import qreflector_teacher_pilot_v9 as pilot
import qreflector_luna_correction as correction
from PIL import Image


class PilotTests(unittest.TestCase):
    def test_selection_is_frozen_proportional_and_unique(self):
        labels = [{'answer': 'Yes'}] * 512 + [{'answer': 'No'}] * 368
        first = pilot.select_indices(labels)
        self.assertEqual(first, pilot.select_indices(labels))
        self.assertEqual(len(set(first)), 100)
        self.assertEqual(Counter(labels[i]['answer'] for i in first), {'Yes': 58, 'No': 42})

    def test_teacher_comparison_only_changes_model_and_student_never_gets_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('RGB', (12, 24), 'red').save(root / 'before.png')
            Image.new('RGB', (12, 24), 'blue').save(root / 'after.png')
            row = {'messages': [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'V9_USER_CONTEXT'},
                    {'type': 'image', 'image': 'before.png'}, {'type': 'image', 'image': 'after.png'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps({
                    'precondition': 'REFERENCE_PRE', 'postcondition': 'REFERENCE_POST',
                    'reason': 'REFERENCE_REASON', 'answer': 'Yes'})}]}]}
            original = copy.deepcopy(row)
            configs = [{'data': str(root), 'image_max_side': 512, 'teacher_model': model,
                        'teacher_reasoning': 'medium', 'teacher_max_output_tokens': 4096}
                       for model in pilot.TEACHERS]
            with patch.object(correction, 'TEACHER_INSTRUCTIONS', pilot.TEACHER_INSTRUCTIONS):
                requests = [correction.teacher_request(row, 'EXACT_SHARED_DRAFT', cfg) for cfg in configs]
            models = [r.pop('model') for r in requests]
            self.assertEqual(tuple(models), pilot.TEACHERS)
            self.assertEqual(requests[0], requests[1])
            student = correction.student_messages(row)
            self.assertEqual(student, [original['messages'][0]])
            self.assertEqual(row, original)
            self.assertNotIn('REFERENCE_', json.dumps(student))
            self.assertNotIn('EXACT_SHARED_DRAFT', json.dumps(student))
            self.assertIn('REFERENCE_', json.dumps(requests[0]))
            self.assertIn('EXACT_SHARED_DRAFT', json.dumps(requests[0]))


if __name__ == '__main__':
    unittest.main()
