"""Regression checks for the BF16 comparison protocol; no GPU or model load."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import qreflector_sft_bf16 as worker
import run_qreflector_bf16 as pipeline


class BF16ProtocolTests(unittest.TestCase):
    def test_precision_guard_checks_tensors_despite_unsloth_compatibility_flag(self):
        model = SimpleNamespace(is_loaded_in_8bit=True, config=SimpleNamespace(),
                modules=lambda: [],
                named_parameters=lambda: [('model.layers.0.q_proj.weight', SimpleNamespace(ndim=2, dtype='bf16'))])
        with patch.dict(sys.modules, {'torch': SimpleNamespace(bfloat16='bf16')}):
            self.assertFalse(worker.check_bf16_base(model)['quantized'])
            model.modules = lambda: [type('Linear4bit', (), {})()]
            with self.assertRaises(AssertionError):
                worker.check_bf16_base(model)

    def test_bf16_and_4bit_input_bytes_are_identical(self):
        import qreflector_sft_v6 as old
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, color in [('before.png', 'red'), ('after.png', 'blue')]:
                Image.new('RGB', (720, 1560), color).save(root / name)
            row = {'messages': [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Task and action.'},
                 {'type': 'image', 'image': 'before.png'}, {'type': 'image', 'image': 'after.png'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"answer":"Yes"}'}]}]}
            saved = copy.deepcopy(row)
            a, b = old.vlm_sample(row, root, 512), worker.vlm_sample(row, root, 512)
            self.assertEqual(a['messages'], b['messages'])
            self.assertEqual([x.tobytes() for x in a['images']], [x.tobytes() for x in b['images']])
            self.assertEqual(row, saved)
            self.assertEqual(worker.PROCESSOR_BASE, old.BASE)

    def test_serial_plan_uses_only_gpu4_and_no_adapter_for_baseline(self):
        stages = pipeline.commands(ROOT / 'artifacts/bf16-plan-test')
        self.assertEqual([x[0] for x in stages],
                         ['base-qwen3-vl-8b-bf16', 'train', 'checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final'])
        self.assertNotIn('--adapter', stages[0][1])
        for name, argv in stages:
            self.assertEqual(argv[argv.index('--gpu-uuid') + 1], worker.GPU_UUID)
            self.assertEqual(argv[argv.index('--memory-cap-gib') + 1], '40')
            self.assertNotIn('--smoke-test', argv)
            if name not in ('train', 'base-qwen3-vl-8b-bf16'):
                self.assertEqual(Path(argv[argv.index('--adapter') + 1]).name, name)

    def test_report_verification_rejects_label_or_prompt_drift(self):
        targets = [{'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Same task'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"answer":"Yes"}'}]}]}] * 1082
        rows = [{'sample_id': i + 1, 'expected': 'yes', 'predicted': 'yes', 'correct': True,
                 'prompt_sha256': hashlib.sha256(b'Same task').hexdigest()} for i in range(1082)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.jsonl'
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            self.assertEqual(len(pipeline.verify_predictions(path, targets)), 1082)
            rows[0]['prompt_sha256'] = 'different'
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaises(AssertionError):
                pipeline.verify_predictions(path, targets)


if __name__ == '__main__':
    unittest.main()
