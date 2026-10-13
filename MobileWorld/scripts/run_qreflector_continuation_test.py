#!/usr/bin/env python3
"""Evaluate both fixed continuation checkpoints on the unchanged 1082-item test."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from qreflector_continue_bf16 import DATA, GPU_UUID, ROOT, SOURCE, digest, read_rows, validate_dataset
from run_qreflector_bf16 import verify_predictions
from eval_gui_reflection import gpu_snapshot, save_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    parent = args.run_dir.resolve()
    assert ROOT / 'artifacts/training' in parent.parents
    run = parent / 'test-evaluation'
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((parent / 'training/status.json').read_text())['cumulative_step'] == 555
    validate_dataset(DATA, images=True)
    worker = ROOT / 'scripts/qreflector_sft_bf16.py'
    previous = SOURCE.parents[1]
    old_plan = json.loads((previous / 'plan.json').read_text())
    assert digest(worker) == old_plan['scripts_sha256'][str(worker)]
    checkpoints = [(444, parent / 'training/checkpoint-111'),
                   (555, parent / 'training/checkpoint-222')]
    test = read_rows(DATA / 'test.jsonl')
    original_rows = verify_predictions(previous / 'evaluations/final/predictions.jsonl', test)
    plan = {'test_sha256': digest(DATA / 'test.jsonl'), 'test_size': 1082,
        'gpu_uuid': GPU_UUID, 'memory_cap_gib': 40, 'worker_sha256': digest(worker),
        'checkpoints': [{'cumulative_step': s, 'adapter': str(p),
                        'sha256': digest(p / 'adapter_model.safetensors')} for s, p in checkpoints],
        'reference_step_333': str(previous / 'evaluations/final/predictions.jsonl'),
        'reference_predictions_sha256': digest(previous / 'evaluations/final/predictions.jsonl'),
        'score': 'Final Yes/No only; invalid counts as incorrect',
        'policy': 'Evaluate both predeclared checkpoints; only start/wait for own children',
        'generation': {'greedy': True, 'max_new_tokens': 1024, 'image_max_side': 512}}
    if (run / 'plan.json').exists():
        assert json.loads((run / 'plan.json').read_text()) == plan
    else:
        save_json(run / 'plan.json', plan)
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    summaries = [dict(summarize(original_rows), cumulative_step=333)]
    completed = []
    for step, adapter in checkpoints:
        output = run / f'step-{step}'
        status_path = output / 'status.json'
        done = status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete'
        if not done:
            while True:
                free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
                if free >= 48:
                    break
                save_json(run / 'status.json', {'status': 'waiting_for_gpu_headroom',
                    'step': step, 'free_gib': free, 'completed_steps': completed})
                time.sleep(30)
            argv = [str(ROOT / 'artifacts/envs/qreflector-native/bin/python'), str(worker), 'evaluate',
                '--output', str(output), '--adapter', str(adapter), '--gpu-uuid', GPU_UUID,
                '--memory-cap-gib', '40', '--limit', '1082', '--image-max-side', '512',
                '--max-new-tokens', '1024', '--resume-results']
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': GPU_UUID, 'PYTHONUNBUFFERED': '1',
                        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
            with (run / f'step-{step}.log').open('a') as log:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                save_json(run / 'status.json', {'status': 'running', 'step': step,
                    'runner_pid': os.getpid(), 'child_pid': child.pid, 'completed_steps': completed,
                    'started_at_utc': datetime.now(timezone.utc).isoformat()})
                code = child.wait()
            if code:
                save_json(run / 'status.json', {'status': 'failed', 'step': step,
                    'returncode': code, 'completed_steps': completed})
                raise SystemExit(code)
        rows = verify_predictions(output / 'predictions.jsonl', test)
        assert all(a['input_tokens'] == b['input_tokens'] for a, b in zip(rows, original_rows))
        summaries.append(dict(summarize(rows), cumulative_step=step))
        completed.append(step)
        save_json(run / 'test_results.json', {'results': summaries, 'test_sha256': plan['test_sha256'],
            'complete': len(completed) == 2, 'score': plan['score']})
    with (run / 'test_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['cumulative_step', 'total', 'correct', 'accuracy',
            'success_recognition_rate', 'failure_recognition_rate', 'invalid_outputs'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    save_json(run / 'status.json', {'status': 'complete', 'completed_steps': completed,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
