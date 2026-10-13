import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from qreflector_sft_v8 import digest, validate_test_selection
from run_qreflector_v8 import choose_best


class ValidationSelectionTests(unittest.TestCase):
    def entries(self):
        return [dict(epoch=e, split='val', total=71, correct=50) for e in range(1, 11)]

    def test_selects_accuracy_winner_with_earliest_epoch_tie(self):
        entries = self.entries()
        entries[5]['correct'] = entries[8]['correct'] = 59
        self.assertEqual(choose_best(list(reversed(entries)))['epoch'], 6)

    def test_requires_all_ten_distinct_complete_validation_epochs(self):
        with self.assertRaises(ValueError):
            choose_best(self.entries()[:9])
        entries = self.entries()
        entries[-1]['epoch'] = 9
        with self.assertRaises(ValueError):
            choose_best(entries)
        entries = self.entries()
        entries[-1]['total'] = 70
        with self.assertRaises(ValueError):
            choose_best(entries)

    def test_rejects_test_results_for_checkpoint_selection(self):
        entries = self.entries()
        entries[0].update(split='test', total=1082, correct=900)
        with self.assertRaises(ValueError):
            choose_best(entries)

    def test_test_worker_requires_selection_before_loading_test(self):
        with self.assertRaises(ValueError):
            validate_test_selection(Namespace(selection_file=None))

    def test_test_worker_rejects_wrong_adapter_or_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            adapter = base / 'checkpoint-660'
            adapter.mkdir()
            weights = adapter / 'adapter_model.safetensors'
            weights.write_bytes(b'selected-weights')
            manifest = base / 'DATASET_MANIFEST.json'
            manifest.write_text('{}')
            selection = base / 'selection.json'
            selection.write_text(json.dumps({
                'selection_split': 'val', 'validation_epochs': list(range(1, 11)),
                'test_used_for_selection': False, 'dataset_manifest_sha256': digest(manifest),
                'adapter': str(adapter), 'adapter_sha256': digest(weights)}))
            args = Namespace(selection_file=selection, adapter=adapter, data=base)
            validate_test_selection(args)
            args.adapter = base / 'checkpoint-770'
            with self.assertRaises(AssertionError):
                validate_test_selection(args)
            args.adapter = adapter
            weights.write_bytes(b'different-weights')
            with self.assertRaises(AssertionError):
                validate_test_selection(args)


if __name__ == '__main__':
    unittest.main()
