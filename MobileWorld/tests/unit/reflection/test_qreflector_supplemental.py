import fcntl
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import run_qreflector_supplemental_test as supplemental


class SupplementalQueueTests(unittest.TestCase):
    def primary(self, base, state='complete'):
        (base / 'test').mkdir()
        (base / 'runner.lock').touch()
        (base / 'status.json').write_text(json.dumps({'status': state}))
        (base / 'test/status.json').write_text(json.dumps({'status': 'complete'}))

    def test_does_not_start_while_primary_pipeline_still_running(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.primary(base, 'running')
            self.assertFalse(supplemental.predecessor_ready(base))

    def test_waits_for_primary_lock_release_even_after_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.primary(base)
            with (base / 'runner.lock').open('r') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(supplemental.predecessor_ready(base))
            self.assertTrue(supplemental.predecessor_ready(base))

    def test_failed_primary_never_allows_supplemental_gpu_start(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.primary(base, 'failed')
            self.assertFalse(supplemental.predecessor_ready(base))

    def test_reuses_exact_evaluator_and_restores_process_local_gate_on_failure(self):
        plan = {'dataset': '/tmp/data', 'adapter': '/tmp/checkpoint-770',
                'gpu_uuid': supplemental.worker.GPU_UUID, 'memory_cap_gib': 32,
                'image_max_side': 512, 'max_new_tokens': 1024}
        original = supplemental.worker.validate_test_selection
        observed = []

        def evaluation(args):
            observed.append((args.split, args.limit, args.adapter.name, args.max_new_tokens))
            self.assertIs(supplemental.worker.validate_test_selection, supplemental.supplemental_gate)
            raise RuntimeError('simulated inference error')

        with patch.object(supplemental, 'verify_supplement', return_value=plan), \
                patch.object(supplemental.worker, 'evaluate', side_effect=evaluation):
            with self.assertRaisesRegex(RuntimeError, 'simulated inference error'):
                supplemental.evaluate_supplement(Path('/tmp/output'))
        self.assertEqual(observed, [('test', 1082, 'checkpoint-770', 1024)])
        self.assertIs(supplemental.worker.validate_test_selection, original)


if __name__ == '__main__':
    unittest.main()
