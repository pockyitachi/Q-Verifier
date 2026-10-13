"""CPU checks for the frozen SFT/evaluation protocol, without loading a model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import qreflector_sft_v6 as worker
import run_qreflector_sft_v6 as pipeline


class SFTProtocolTests(unittest.TestCase):
    def test_label_parser_does_not_use_rationale_words(self):
        self.assertEqual(worker.parse_verdict('{"reason":"Yes is unsupported here", "answer":"No"}'), 'no')
        self.assertIsNone(worker.parse_verdict('The result could be Yes or No.'))
        self.assertIsNone(worker.parse_verdict('{"reason":"No visible change"}'))
        good = {'precondition':'Input is available.', 'postcondition':'Input opens.', 'reason':'It opened.', 'answer':'Yes'}
        self.assertTrue(worker.strict_structure(json.dumps(good)))
        self.assertFalse(worker.strict_structure('{"answer":"Yes"}'))

    def test_images_and_target_survive_conversion_without_mutating_source(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, color in [('before.png', 'red'), ('after.png', 'blue')]:
                Image.new('RGB', (720, 1560), color).save(root / name)
            row = {'messages':[
                {'role':'user', 'content':[{'type':'text','text':'Task, action and history.'},
                    {'type':'image','image':'before.png'}, {'type':'image','image':'after.png'}]},
                {'role':'assistant','content':[{'type':'text','text':'{"answer":"No"}'}]}]}
            previous = copy.deepcopy(row)
            converted = worker.vlm_sample(row, root, 512)
            self.assertEqual(row, previous)
            self.assertEqual(converted['messages'][1], row['messages'][1])
            self.assertEqual([max(x.size) for x in converted['images']], [512, 512])
            self.assertEqual(converted['images'][0].getpixel((0,0)), (255,0,0))
            self.assertEqual(converted['images'][1].getpixel((0,0)), (0,0,255))
            (root/'after.png').unlink()
            with self.assertRaises(FileNotFoundError):
                worker.vlm_sample(row, root, 512)

    def test_six_evaluations_have_matched_inputs_and_separate_processes(self):
        plan = pipeline.commands(ROOT / 'artifacts/test-plan-only')
        self.assertEqual([x[0] for x in plan], ['train','checkpoint-100','checkpoint-200','checkpoint-300','final','base-qwen3-vl-8b','gui-reflection-8b'])
        for name, argv in plan:
            self.assertEqual(argv[argv.index('--gpu-uuid')+1], worker.GPU0_UUID)
            self.assertEqual(argv[argv.index('--image-max-side')+1], '512')
            if name != 'train':
                self.assertEqual(argv[argv.index('--max-new-tokens')+1], '1024')
        for name, argv in plan[1:5]:
            self.assertEqual(Path(argv[argv.index('--adapter')+1]).name, name)
        self.assertNotIn('--adapter', plan[5][1])

    def test_invalid_verdicts_count_as_errors_not_negative_predictions(self):
        summary = worker.summarize([
            {'expected':'yes','predicted':'yes','correct':True},
            {'expected':'yes','predicted':None,'correct':False},
            {'expected':'no','predicted':'no','correct':True},
            {'expected':'no','predicted':'yes','correct':False},
        ])
        self.assertEqual(summary['accuracy'], 0.5)
        self.assertEqual(summary['success_recognition_rate'], 0.5)
        self.assertEqual(summary['failure_recognition_rate'], 0.5)
        self.assertEqual(summary['invalid_outputs'], 1)
        self.assertEqual(summary['confusion']['yes']['no'], 0)


if __name__ == '__main__':
    unittest.main()
