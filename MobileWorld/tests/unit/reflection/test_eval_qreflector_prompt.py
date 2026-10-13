import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))

from eval_qreflector_prompt import render_user


class PromptReplacementTests(unittest.TestCase):
    def setUp(self):
        self.original = 'Frozen instructions\n=== Context ===\n{context}\nEND\n'
        self.replacement = 'Check conditions before verdict.\n{context}\nReturn four fields.\n'
        self.context = {
            'goal': 'Search for literal {context} and "END".',
            'history': ['A note containing === Context ==='],
            'action': {'type': 'tap', 'arguments': {'x': 25, 'y': 40}},
        }
        self.raw = json.dumps(self.context, indent=2)
        self.row = {'messages': [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': self.original.format(context=self.raw)},
                {'type': 'image', 'image': '../screens/before.png'},
                {'type': 'image', 'image': '../screens/after.png'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'REFERENCE_ONLY_SENTINEL'}]},
        ]}

    def test_preserves_context_images_and_source_without_reference_leakage(self):
        before = copy.deepcopy(self.row)
        user, record = render_user(self.row, self.original, self.replacement)
        self.assertEqual(self.row, before)
        self.assertEqual(user['content'][0]['text'], self.replacement.format(context=self.raw))
        self.assertEqual(user['content'][1:], before['messages'][0]['content'][1:])
        self.assertEqual(record['image_paths'], ['../screens/before.png', '../screens/after.png'])
        self.assertNotIn('REFERENCE_ONLY_SENTINEL', json.dumps(user))
        self.assertNotEqual(record['prompt_sha256'], record['original_prompt_sha256'])

    def test_rejects_unexpected_context_fields(self):
        self.context['answer'] = 'No'
        self.row['messages'][0]['content'][0]['text'] = self.original.format(context=json.dumps(self.context))
        with self.assertRaises(ValueError):
            render_user(self.row, self.original, self.replacement)

    def test_rejects_unknown_or_duplicate_placeholders(self):
        for replacement in ['{context} {answer}', '{context} {context}', 'no placeholder']:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                render_user(self.row, self.original, replacement)

    def test_rejects_nonmatching_source_or_missing_image(self):
        self.row['messages'][0]['content'][0]['text'] += 'extra'
        with self.assertRaises(ValueError):
            render_user(self.row, self.original, self.replacement)
        self.row['messages'][0]['content'][0]['text'] = self.original.format(context=self.raw)
        self.row['messages'][0]['content'].pop()
        with self.assertRaises(ValueError):
            render_user(self.row, self.original, self.replacement)


if __name__ == '__main__':
    unittest.main()
