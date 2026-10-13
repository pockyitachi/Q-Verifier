#!/usr/bin/env python3
"""Serial GPU-4 BF16 baseline, LoRA SFT and four predeclared evaluations.

Only starts its own subprocesses and waits for their normal exits. Never sends
signals to existing processes or modifies GPU settings or running services.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from qreflector_sft_bf16 import BASE, DATA, GPU_UUID, PROCESSOR_BASE, ROOT, digest, read_rows, validate_dataset
from eval_gui_reflection import gpu_snapshot, save_json, summarize

PREVIOUS = ROOT / 'artifacts/training/qreflector_sft_v6_20261002_gpu0'
CHECKPOINTS = ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final']


def commands(run):
    worker = ROOT / 'scripts/qreflector_sft_bf16.py'
    shared = ['--data', str(DATA), '--gpu-uuid', GPU_UUID,
              '--memory-cap-gib', '40', '--image-max-side', '512']

    def evaluation(name, adapter=None):
        argv = [str(ROOT / 'artifacts/envs/qreflector-native/bin/python'), str(worker), 'evaluate',
                '--output', str(run / 'evaluations' / name), *shared,
                '--max-new-tokens', '1024', '--resume-results']
        if adapter:
            argv += ['--adapter', str(adapter)]
        return name, argv

    result = [evaluation('base-qwen3-vl-8b-bf16')]
    result.append(('train', [str(ROOT / 'artifacts/envs/qreflector-sft/bin/python'), str(worker), 'train',
                             '--output', str(run / 'training'), *shared]))
    result.extend(evaluation(name, run / 'training' / name) for name in CHECKPOINTS)
    return result


def verify_predictions(path, expected_rows):
    import hashlib
    rows = read_rows(path)
    assert len(rows) == 1082 and [r['sample_id'] for r in rows] == list(range(1, 1083))
    for row, target in zip(rows, expected_rows):
        answer = json.loads(target['messages'][1]['content'][0]['text'])['answer'].lower()
        prompt = '\n'.join(c['text'] for c in target['messages'][0]['content'] if c['type'] == 'text')
        assert row['expected'] == answer
        assert row['correct'] == (row['predicted'] == answer)
        assert row['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    return rows


def report(run):
    summaries = []
    test = read_rows(DATA / 'test.jsonl')
    old_reference = verify_predictions(PREVIOUS / 'evaluations/base-qwen3-vl-8b/predictions.jsonl', test)
    for name, _ in commands(run):
        if name == 'train':
            continue
        path = run / 'evaluations' / name / 'predictions.jsonl'
        rows = verify_predictions(path, test)
        assert all(a['input_tokens'] == b['input_tokens'] for a, b in zip(rows, old_reference)), name
        s = summarize(rows)
        s.update({'model': name, 'precision': 'BF16',
                  'balanced_accuracy': (s['success_recognition_rate'] + s['failure_recognition_rate']) / 2,
                  'mean_latency_seconds': sum(r['latency_seconds'] for r in rows) / len(rows),
                  'predictions': str(path)})
        summaries.append(s)
    result = {'results': summaries, 'test_sha256': digest(DATA / 'test.jsonl'),
              'matched_prompts_labels_and_input_token_counts': True,
              'score': 'Final Yes/No only; invalid verdicts count as incorrect.',
              'checkpoint_policy': 'Report 100, 200, 300 and final; no test-based selection or tuning.',
              'previous_results': str(PREVIOUS / 'six_results.json')}
    save_json(run / 'bf16_results.json', result)
    previous = json.loads((PREVIOUS / 'six_results.json').read_text())['results']
    comparison = [dict(s, precision='BF16' if s['model'] == 'gui-reflection-8b' else '4-bit') for s in previous]
    comparison += summaries
    columns = ['precision', 'model', 'total', 'correct', 'accuracy', 'success_recognition_rate',
               'failure_recognition_rate', 'balanced_accuracy', 'invalid_outputs', 'mean_latency_seconds']
    with (run / 'precision_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparison)
    lines = ['Q-Reflector v6 BF16 comparison', '',
             '882 train / 71 validation / 1082 test. Final Yes/No scoring only.',
             'Matched user text, images, processor, greedy decoding and token budget.',
             'All four fixed checkpoints reported; no test-based checkpoint selection.', '',
             'Precision | Model | Accuracy | Success recognition | Failure recognition | Invalid']
    for s in comparison:
        lines.append(f"{s['precision']} | {s['model']} | {100*s['accuracy']:.2f}% | "
                     f"{100*s['success_recognition_rate']:.2f}% | {100*s['failure_recognition_rate']:.2f}% | {s['invalid_outputs']}")
    (run / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    return result


def wait_for_headroom(run, stage, completed):
    while True:
        free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
        if free >= 48:
            return
        save_json(run / 'status.json', {'status': 'waiting_for_gpu_headroom', 'stage': stage,
                  'free_gib': free, 'required_free_gib': 48, 'runner_pid': os.getpid(),
                  'completed_stages': completed, 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
        print(f'Waiting for 48 GiB free on GPU 4; currently {free:.1f}. Existing processes untouched.', flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert ROOT / 'artifacts' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = validate_dataset(DATA, images=True)
    source = json.loads((BASE.parent / 'source.json').read_text())
    verification = json.loads((BASE.parent / 'verified.json').read_text())
    assert source['revision'] == verification['revision']
    weight_comparison = json.loads((BASE.parent / 'weight_comparison.json').read_text())
    assert not weight_comparison['mismatches']
    for item in verification['files']:
        assert digest(BASE / item['path']) == item['sha256'], item['path']
    assert not json.loads((BASE / 'config.json').read_text()).get('quantization_config')
    runtime = json.loads((PREVIOUS / 'runtime_environment.json').read_text())
    save_json(run / 'runtime_environment.json', runtime)
    scripts = [Path(__file__), ROOT / 'scripts/qreflector_sft_bf16.py', ROOT / 'scripts/eval_gui_reflection.py']
    processor_files = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                       'preprocessor_config.json', 'generation_config.json']
    plan = {'dataset': str(DATA), 'frozen_manifest_sha256': digest(DATA / 'FROZEN.json'),
            'splits': frozen['splits'], 'base': str(BASE), 'base_source': source,
            'base_verified_manifest_sha256': digest(BASE.parent / 'verified.json'),
            'weight_comparison_to_previous_base': weight_comparison,
            'processor_source': str(PROCESSOR_BASE),
            'processor_sha256': {name: digest(PROCESSOR_BASE / name) for name in processor_files},
            'gpu_uuid': GPU_UUID, 'memory_cap_gib': 40, 'minimum_startup_free_gib': 48,
            'scripts_sha256': {str(p): digest(p) for p in scripts}, 'runtime_environment': runtime,
            'training': {'epochs': 3, 'batch_size': 1, 'grad_accum': 8, 'steps': 333,
                         'learning_rate': 2e-4, 'rank': 16, 'alpha': 16, 'optimizer': 'adamw_8bit',
                         'weights': 'unquantized BF16', 'assistant_only_loss': True,
                         'expected_trainable_parameters': 43646976},
            'evaluation': {'test_size': 1082, 'total_new_predictions': 5410,
                           'image_max_side': 512, 'max_new_tokens': 1024, 'greedy': True,
                           'metric': 'Final Yes/No only'},
            'stages': [{'name': name, 'argv': argv} for name, argv in commands(run)],
            'policy': 'Spawn and wait for own children only; never signal other processes, reset GPUs or modify services.'}
    plan_path = run / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Existing run plan differs'
    else:
        save_json(plan_path, plan)
    if args.prepare_only:
        print(json.dumps({'prepared': str(plan_path), 'stages': len(plan['stages'])}), flush=True)
        return
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    (run / 'logs').mkdir(exist_ok=True)
    completed = []
    for name, argv in commands(run):
        stage_dir = run / 'training' if name == 'train' else run / 'evaluations' / name
        status_path = stage_dir / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') == 'complete':
            if name == 'train':
                assert status['step'] == 333
                assert all((stage_dir / n / 'adapter_model.safetensors').is_file() for n in CHECKPOINTS)
            else:
                assert len(read_rows(stage_dir / 'predictions.jsonl')) == 1082
            completed.append(name)
            continue
        if name == 'train' and stage_dir.exists():
            checkpoints = sorted(stage_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            checkpoints = [p for p in checkpoints if (p / 'trainer_state.json').is_file() and (p / 'optimizer.pt').is_file()]
            if checkpoints:
                argv += ['--resume', str(checkpoints[-1])]
        wait_for_headroom(run, name, completed)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': GPU_UUID, 'PYTHONUNBUFFERED': '1',
                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        if name == 'train':
            env.update(runtime)
        with (run / 'logs' / f'{name}.log').open('a') as log:
            print(f'Starting {name}', flush=True)
            try:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
            except OSError as error:
                save_json(run / 'status.json', {'status': 'failed_launch', 'stage': name,
                          'error': str(error), 'completed_stages': completed})
                raise
            save_json(run / 'status.json', {'status': 'running', 'stage': name,
                      'runner_pid': os.getpid(), 'child_pid': child.pid, 'completed_stages': completed,
                      'started_at_utc': datetime.now(timezone.utc).isoformat()})
            code = child.wait()
        if code:
            save_json(run / 'status.json', {'status': 'failed', 'stage': name, 'returncode': code,
                      'completed_stages': completed, 'log': str(run / 'logs' / f'{name}.log')})
            raise SystemExit(code)
        completed.append(name)
        print(f'Completed {name}', flush=True)
    result = report(run)
    save_json(run / 'status.json', {'status': 'complete', 'completed_stages': completed,
              'results': str(run / 'bf16_results.json'), 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
