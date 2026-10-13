#!/usr/bin/env python3
"""Run a prepared OPSD pilot after its smoke test, then evaluate its final adapter."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eval_gui_reflection import summarize
from qreflector_opsd import ROOT, digest, load_plan, read_json, write_json
from run_qreflector_bf16 import verify_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke-run', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--baseline-predictions', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    smoke = args.smoke_run.resolve()
    plan, _ = load_plan(run)
    smoke_plan, _ = load_plan(smoke)
    config = plan['config']
    assert all(config[k] == smoke_plan['config'][k] for k in config if k != 'max_steps')
    assert smoke_plan['optimizer_steps'] == 1 and plan['optimizer_steps'] == 111
    python = str(ROOT / 'artifacts/envs/qreflector-native/bin/python')
    evaluation = run / 'test-final'
    commands = [
        ('train', [python, str(ROOT / 'scripts/qreflector_opsd.py'), 'train',
                   '--run-dir', str(run), '--gpu-uuid', args.gpu_uuid,
                   '--memory-cap-gib', '40', '--reserve-free-gib', '8']),
        ('test-final', [python, str(ROOT / 'scripts/qreflector_sft_bf16.py'), 'evaluate',
                        '--data', config['data'], '--adapter', str(run / 'final'),
                        '--output', str(evaluation), '--gpu-uuid', args.gpu_uuid,
                        '--memory-cap-gib', '40', '--image-max-side', '512',
                        '--limit', '1082', '--max-new-tokens', '1024', '--resume-results']),
    ]
    sources = [Path(__file__), ROOT / 'scripts/qreflector_sft_bf16.py',
               ROOT / 'scripts/eval_gui_reflection.py', ROOT / 'scripts/run_qreflector_bf16.py']
    baseline = args.baseline_predictions.resolve()
    runner_plan = {
        'training_plan_sha256': digest(run / 'plan.json'), 'smoke_run': str(smoke),
        'gpu_uuid': args.gpu_uuid, 'memory_cap_gib': 40, 'reserve_free_gib': 8,
        'stages': [{'name': name, 'argv': argv} for name, argv in commands],
        'script_sha256': {str(path): digest(path) for path in sources},
        'baseline_predictions': str(baseline), 'baseline_sha256': digest(baseline),
        'test_sha256': digest(Path(config['data']) / 'test.jsonl'),
        'evaluation_policy': 'One predetermined final adapter; no test-based checkpoint selection.',
        'score': 'Final Yes/No only; invalid verdicts count as wrong.',
        'initialization': 'Original epoch-6 adapter with a fresh OPSD optimizer; smoke weights are discarded.',
        'process_policy': 'Only spawn and wait for own children; never signal existing processes.',
    }
    with (run / '.runner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = run / 'runner_plan.json'
        if destination.exists():
            assert read_json(destination) == runner_plan, 'Runner plan changed'
        else:
            write_json(destination, runner_plan)
        if args.prepare_only:
            print(json.dumps({'prepared': str(run), 'stages': ['train', 'test-final']}))
            return
        smoke_status = read_json(smoke / 'status.json')
        assert smoke_status['status'] == 'complete' and smoke_status['step'] == 1
        assert smoke_status['teacher_unchanged'] and smoke_status['gpu_uuid'] == args.gpu_uuid
        smoke_metrics = read_json(smoke / 'metrics.jsonl')
        assert smoke_metrics['batch_examples'] == 8 and smoke_metrics['gradient_norm'] > 0
        assert digest(smoke / 'final/adapter_model.safetensors') != plan['pinned_files'][
            str(Path(config['adapter']) / 'adapter_model.safetensors')]
        logs = run / 'logs'
        logs.mkdir(exist_ok=True)
        completed = []
        stage = 'startup'
        try:
            for stage, argv in commands:
                stage_status = run / 'status.json' if stage == 'train' else evaluation / 'status.json'
                if stage_status.exists() and read_json(stage_status).get('status') == 'complete':
                    completed.append(stage)
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONUNBUFFERED='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                with (logs / f'{stage}.log').open('a') as log:
                    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    write_json(run / 'pipeline_status.json', {
                        'status': 'running', 'stage': stage, 'runner_pid': os.getpid(),
                        'child_pid': child.pid, 'completed_stages': completed,
                        'started_at_utc': datetime.now(UTC).isoformat(),
                    })
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{stage} exited with code {code}; see {logs / (stage + ".log")}')
                assert read_json(stage_status)['status'] == 'complete'
                if stage == 'train':
                    assert read_json(stage_status)['step'] == plan['optimizer_steps']
                completed.append(stage)
            stage = 'report'
            test = [json.loads(line) for line in (Path(config['data']) / 'test.jsonl').read_text().splitlines()]
            before = verify_predictions(baseline, test)
            after = verify_predictions(evaluation / 'predictions.jsonl', test)
            assert all(a['input_tokens'] == b['input_tokens'] for a, b in zip(before, after))
            initial, final = summarize(before), summarize(after)
            write_json(run / 'comparison.json', {
                'baseline_epoch6': initial, 'opsd_final': final,
                'accuracy_delta_percentage_points': 100 * (final['accuracy'] - initial['accuracy']),
                'matched_prompts_labels_input_token_counts': True,
                'score': runner_plan['score'], 'test_sha256': runner_plan['test_sha256'],
                'corrected_errors': sum(not a['correct'] and b['correct'] for a, b in zip(before, after)),
                'new_errors': sum(a['correct'] and not b['correct'] for a, b in zip(before, after)),
            })
            load_plan(run)  # Recheck that the source checkpoint and frozen train data are unchanged.
            assert digest(baseline) == runner_plan['baseline_sha256']
            write_json(run / 'pipeline_status.json', {
                'status': 'complete', 'completed_stages': completed,
                'completed_at_utc': datetime.now(UTC).isoformat(),
            })
        except BaseException as error:
            write_json(run / 'pipeline_status.json', {
                'status': 'failed', 'stage': stage, 'error': f'{type(error).__name__}: {error}',
                'completed_stages': completed,
            })
            raise


if __name__ == '__main__':
    main()
