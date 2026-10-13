#!/usr/bin/env python3
"""Serial epochs 6-10 BF16 continuation and five fixed 1082-item test evaluations."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from qreflector_bf16_epoch10 import DATA, GPU_UUID, ROOT, SOURCE, digest, read_rows, validate_dataset
from run_qreflector_bf16 import verify_predictions
from eval_gui_reflection import gpu_snapshot, save_json, summarize

ORIGINAL = ROOT / 'artifacts/training/qreflector_bf16_v6_20261004_gpu4'
PREVIOUS = SOURCE.parents[1]


def stages(run):
    common = ['--gpu-uuid', GPU_UUID, '--memory-cap-gib', '40', '--image-max-side', '512']
    result = [('train', [str(ROOT / 'artifacts/envs/qreflector-sft/bin/python'),
        str(ROOT / 'scripts/qreflector_bf16_epoch10.py'), '--output', str(run / 'training'), *common])]
    for epoch in range(6, 11):
        name = f'epoch-{epoch}-step-{epoch * 111}'
        result.append((name, [str(ROOT / 'artifacts/envs/qreflector-native/bin/python'),
            str(ROOT / 'scripts/qreflector_sft_bf16.py'), 'evaluate',
            '--output', str(run / 'evaluations' / name), '--adapter', str(run / 'training' / name),
            '--limit', '1082', '--max-new-tokens', '1024', '--resume-results', *common]))
    return result


def report(run, complete=False):
    test = read_rows(DATA / 'test.jsonl')
    reference = verify_predictions(ORIGINAL / 'evaluations/final/predictions.jsonl', test)
    entries = [(3, reference)]
    for epoch in (4, 5):
        entries.append((epoch, verify_predictions(PREVIOUS / f'test-evaluation/step-{epoch * 111}/predictions.jsonl', test)))
    for epoch in range(6, 11):
        directory = run / 'evaluations' / f'epoch-{epoch}-step-{epoch * 111}'
        status = directory / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'complete':
            continue
        rows = verify_predictions(directory / 'predictions.jsonl', test)
        assert all(a['input_tokens'] == b['input_tokens'] for a, b in zip(reference, rows))
        entries.append((epoch, rows))
    if complete:
        assert len(entries) == 8
    results = [dict(summarize(rows), epoch=epoch, cumulative_step=epoch * 111) for epoch, rows in entries]
    save_json(run / 'test_results.json', {'complete': complete, 'results': results,
        'test_sha256': digest(DATA / 'test.jsonl'), 'score': 'Final Yes/No only; invalid is wrong',
        'training_protocol': 'Staged continuation with LR restarts, not uninterrupted ten-epoch training',
        'checkpoint_policy': 'All five predeclared epoch checkpoints evaluated and reported'})
    with (run / 'test_comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'cumulative_step', 'total', 'correct', 'accuracy',
            'success_recognition_rate', 'failure_recognition_rate', 'invalid_outputs'], extrasaction='ignore')
        w.writeheader()
        w.writerows(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    run = args.run_dir.resolve()
    assert ROOT / 'artifacts/training' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_dataset(DATA, images=True)
    assert json.loads((PREVIOUS / 'training/status.json').read_text())['cumulative_step'] == 555
    source_names = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json']
    source_hashes = {n: digest(SOURCE / n) for n in source_names}
    old_plan = json.loads((ORIGINAL / 'plan.json').read_text())
    for path, sha in old_plan['scripts_sha256'].items():
        assert digest(Path(path)) == sha, 'Original protocol changed'
    runtime = json.loads((ORIGINAL / 'runtime_environment.json').read_text())
    plan = {'source': str(SOURCE), 'source_sha256': source_hashes,
        'frozen_manifest_sha256': digest(DATA / 'FROZEN.json'), 'test_sha256': digest(DATA / 'test.jsonl'),
        'source_epochs': 5, 'additional_epochs': 5, 'additional_optimizer_steps': 555,
        'cumulative_final_step': 1110, 'save_each_epoch': True,
        'optimizer_state_restored': True, 'scheduler_restarted': True,
        'learning_rate': 2e-5, 'scheduler': 'cosine', 'warmup_ratio': 0.05,
        'seed': 42, 'data_seed': 47, 'batch_size': 1, 'gradient_accumulation_steps': 8,
        'evaluation_epochs': list(range(6, 11)), 'test_samples_each': 1082,
        'generation': {'greedy': True, 'max_new_tokens': 1024, 'image_max_side': 512},
        'gpu_uuid': GPU_UUID, 'memory_cap_gib': 40, 'minimum_startup_free_gib': 48,
        'runtime_environment': runtime,
        'scripts_sha256': {str(x): digest(x) for x in [Path(__file__), ROOT / 'scripts/qreflector_bf16_epoch10.py']},
        'stages': [{'name': n, 'argv': a} for n, a in stages(run)],
        'policy': 'Only spawn/wait for own children; never modify existing processes or GPU settings'}
    if (run / 'plan.json').exists():
        assert json.loads((run / 'plan.json').read_text()) == plan
    else:
        save_json(run / 'plan.json', plan)
    if args.prepare_only:
        print(json.dumps({'prepared': str(run), 'epochs': [6, 7, 8, 9, 10], 'new_test_predictions': 5410}))
        return
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    (run / 'logs').mkdir(exist_ok=True)
    completed = []
    for name, argv in stages(run):
        stage_dir = run / 'training' if name == 'train' else run / 'evaluations' / name
        status_path = stage_dir / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') == 'complete':
            if name == 'train':
                assert status['cumulative_step'] == 1110
            else:
                assert len(read_rows(stage_dir / 'predictions.jsonl')) == 1082
            completed.append(name)
            continue
        if name == 'train' and status_path.exists():
            raise RuntimeError('Partial training exists; recover explicitly instead of overwriting')
        while True:
            free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
            if free >= 48:
                break
            save_json(run / 'status.json', {'status': 'waiting_for_gpu_headroom',
                'stage': name, 'free_gib': free, 'completed_stages': completed})
            time.sleep(30)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': GPU_UUID, 'PYTHONUNBUFFERED': '1',
                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        if name == 'train':
            env.update(runtime)
        with (run / 'logs' / f'{name}.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            save_json(run / 'status.json', {'status': 'running', 'stage': name,
                'runner_pid': os.getpid(), 'child_pid': child.pid, 'completed_stages': completed,
                'started_at_utc': datetime.now(timezone.utc).isoformat()})
            code = child.wait()
        if code:
            save_json(run / 'status.json', {'status': 'failed', 'stage': name,
                'returncode': code, 'completed_stages': completed})
            raise SystemExit(code)
        completed.append(name)
        if name != 'train':
            report(run)
    report(run, complete=True)
    assert all(digest(SOURCE / n) == sha for n, sha in source_hashes.items())
    save_json(run / 'status.json', {'status': 'complete', 'completed_stages': completed,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
