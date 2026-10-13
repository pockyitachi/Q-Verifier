#!/usr/bin/env python3
"""Run the authorized GPU-0 SFT and six evaluations in strictly serial children.

This runner never signals, resets, or terminates any GPU process. A child's
normal exit unloads its own model before the next stage starts.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from qreflector_sft_v6 import BASE, DATA, ROOT, GPU0_UUID, digest, read_rows, strict_structure, validate_dataset
from eval_gui_reflection import gpu_snapshot, save_json, summarize


def commands(run):
    worker = ROOT / 'scripts/qreflector_sft_v6.py'
    shared = ['--data', str(DATA), '--gpu-uuid', GPU0_UUID, '--memory-cap-gib', '32', '--image-max-side', '512']
    training = run / 'training'
    result = [('train', [str(ROOT / 'artifacts/envs/qreflector-sft/bin/python'), str(worker), 'train',
                         '--output', str(training), *shared])]
    checkpoints = ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final']
    for name in checkpoints + ['base-qwen3-vl-8b']:
        cmd = [str(ROOT / 'artifacts/envs/qreflector-native/bin/python'), str(worker), 'evaluate',
               '--output', str(run / 'evaluations' / name), *shared,
               '--max-new-tokens', '1024', '--resume-results']
        if name in checkpoints:
            cmd += ['--adapter', str(training / name)]
        result.append((name, cmd))
    result.append(('gui-reflection-8b', [str(ROOT / 'artifacts/envs/gui-reflection/bin/python'),
        str(ROOT / 'scripts/eval_gui_reflection.py'), '--data', str(DATA / 'test.jsonl'),
        '--output-dir', str(run / 'evaluations/gui-reflection-8b'), '--gpu-uuid', GPU0_UUID,
        '--memory-cap-gib', '32', '--image-max-side', '512', '--max-new-tokens', '1024', '--resume']))
    return result


def report(run):
    summaries, predictions = [], []
    for name, _ in commands(run)[1:]:
        directory = run / 'evaluations' / name
        rows = read_rows(directory / 'predictions.jsonl')
        assert len(rows) == 1082 and [r['sample_id'] for r in rows] == list(range(1, 1083))
        predictions.append(rows)
        summary = summarize(rows)
        summary.update({'model': name,
            'balanced_accuracy': (summary['success_recognition_rate'] + summary['failure_recognition_rate']) / 2,
            'valid_four_field_json_rate': sum(strict_structure(r['full_response']) for r in rows) / 1082,
            'total_inference_seconds': sum(r['latency_seconds'] for r in rows),
            'mean_latency_seconds': sum(r['latency_seconds'] for r in rows) / 1082,
            'predictions': str(directory / 'predictions.jsonl')})
        summaries.append(summary)
    for i in range(1082):
        assert len({rows[i]['expected'] for rows in predictions}) == 1
        assert len({rows[i]['prompt_sha256'] for rows in predictions}) == 1
    save_json(run / 'six_results.json', {'results': summaries,
        'same_1082_samples_and_prompt_hashes': True,
        'test_sha256': digest(DATA / 'test.jsonl'),
        'checkpoint_policy': 'Four predeclared checkpoints, all reported. No test-based hyperparameter changes.',
        'precision': 'All Qwen evaluations share the same original 4-bit base. GUI-Reflection uses its native BF16 implementation.',
        'input_policy': 'Same frozen user text and 512-pixel longest-side images, followed by each architecture native preprocessing.'})
    columns = ['model', 'total', 'correct', 'accuracy', 'success_recognition_rate',
               'failure_recognition_rate', 'balanced_accuracy', 'false_rejection_rate',
               'false_acceptance_rate', 'invalid_outputs', 'valid_four_field_json_rate',
               'mean_latency_seconds', 'total_inference_seconds']
    with (run / 'six_results.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(summaries)
    text = ['Q-Reflector v6: SFT and six fixed AndroidWorld transition evaluations', '',
            'Dataset: train 882 / val 71 / test 1082. Test labels: Yes 613 / No 469.',
            'Single physical GPU 0. Each child exits before the next model is loaded.',
            'Qwen: original 4-bit base, no adapter for the untrained baseline.',
            'GUI-Reflection: official model wrapper, BF16. This is a matched-input comparison,',
            "not a reproduction of the authors' native benchmark.",
            'Invalid verdicts are counted as incorrect and reported separately.', '',
            'Model | Accuracy | Success recognition | Failure recognition | Invalid | JSON schema']
    for s in summaries:
        text.append(f"{s['model']} | {100*s['accuracy']:.2f}% | {100*s['success_recognition_rate']:.2f}% | "
                    f"{100*s['failure_recognition_rate']:.2f}% | {s['invalid_outputs']} | {100*s['valid_four_field_json_rate']:.2f}%")
    (run / 'REPORT.txt').write_text('\n'.join(text) + '\n')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    run = a.run_dir.resolve()
    assert ROOT / 'artifacts' in run.parents, 'Run directory must be inside this project artifacts'
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = validate_dataset(DATA, images=True)
    plan = {'dataset': str(DATA), 'frozen_manifest_sha256': digest(DATA / 'FROZEN.json'),
        'splits': frozen['splits'], 'gpu_uuid': GPU0_UUID, 'memory_cap_gib': 32,
        'base': str(BASE), 'base_config_sha256': digest(BASE / 'config.json'),
        'base_source': json.loads((BASE.parent / 'source.json').read_text()),
        'gui_source': json.loads((ROOT / 'artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain.download.json').read_text()),
        'scripts_sha256': {str(f): digest(f) for f in [Path(__file__), ROOT / 'scripts/qreflector_sft_v6.py', ROOT / 'scripts/eval_gui_reflection.py']},
        'stages': [{'name': name, 'argv': cmd} for name, cmd in commands(run)],
        'training': {'epochs': 3, 'batch_size': 1, 'grad_accum': 8, 'learning_rate': 2e-4,
            'expected_steps': 333, 'lora_rank': 16, 'assistant_only_loss': True},
        'evaluation': {'samples_per_model': 1082, 'total_predictions': 6492,
            'max_new_tokens': 1024, 'image_max_side': 512, 'greedy': True},
        'process_policy': 'Only spawn and wait for this runner children; no signals to existing processes, no GPU resets, no service changes.',
    }
    path = run / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Existing run plan differs'
    else:
        save_json(path, plan)
    if a.prepare_only:
        print(json.dumps({'prepared': str(path), 'stages': len(plan['stages'])}, indent=2))
        return
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU0_UUID))
    (run / 'logs').mkdir(exist_ok=True)
    completed = []
    for name, argv in commands(run):
        stage_dir = run / 'training' if name == 'train' else run / 'evaluations' / name
        status_file = stage_dir / 'status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get('status') == 'complete':
            if name == 'train':
                assert status['step'] == 333
                assert all((stage_dir / n / 'adapter_model.safetensors').exists() for n in ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final'])
            else:
                assert len(read_rows(stage_dir / 'predictions.jsonl')) == 1082
            completed.append(name)
            continue
        if name == 'train' and stage_dir.exists():
            checkpoints = sorted(stage_dir.glob('checkpoint-*'), key=lambda x: int(x.name.split('-')[-1]))
            checkpoints = [q for q in checkpoints if (q / 'trainer_state.json').is_file() and (q / 'optimizer.pt').is_file()]
            if checkpoints:
                argv += ['--resume', str(checkpoints[-1])]
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': GPU0_UUID, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        runtime_environment = run / 'runtime_environment.json'
        if name == 'train' and runtime_environment.exists():
            env.update(json.loads(runtime_environment.read_text()))
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
            save_json(run / 'status.json', {'status': 'failed', 'stage': name,
                'returncode': code, 'completed_stages': completed, 'log': str(run / 'logs' / f'{name}.log')})
            raise SystemExit(code)
        # A normal child exit unloads CUDA state. No process is killed here.
        completed.append(name)
        print(f'Completed {name}', flush=True)
    summaries = report(run)
    save_json(run / 'status.json', {'status': 'complete', 'completed_stages': completed,
        'results': str(run / 'six_results.json'), 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
