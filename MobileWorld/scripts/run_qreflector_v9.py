#!/usr/bin/env python3
"""Train fifteen epochs, select on all 71 validation rows, test the winner once."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from eval_gui_reflection import gpu_snapshot, parse_verdict, save_json, summarize
from qreflector_sft_v9 import (
    BASE, DATA, EPOCHS, STEPS_PER_EPOCH, GPU_UUID, PROCESSOR_BASE, ROOT, digest, read_rows, validate_dataset,
)

WORKER = ROOT / 'scripts/qreflector_sft_v9.py'
NATIVE = ROOT / 'artifacts/envs/qreflector-native/bin/python'
SFT = ROOT / 'artifacts/envs/qreflector-sft/bin/python'
RUNTIME = ROOT / 'artifacts/training/qreflector_bf16_v6_20261004_gpu4/runtime_environment.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def command(run, name, *, epoch=None):
    shared = ['--data', str(DATA), '--gpu-uuid', GPU_UUID,
              '--memory-cap-gib', '32', '--image-max-side', '512']
    if name in ('smoke', 'train'):
        args = [str(SFT), str(WORKER), 'train', '--output', str(run / name), *shared]
        return args + (['--smoke-test'] if name == 'smoke' else [])
    if name == 'validation':
        assert epoch in range(1, EPOCHS + 1)
        output = run / 'validation' / f'epoch-{epoch}'
        adapter = run / 'train' / f'checkpoint-{epoch * STEPS_PER_EPOCH}'
        extra = ['--split', 'val']
    elif name == 'test':
        selection = json.loads((run / 'selection.json').read_text())
        output, adapter = run / 'test', Path(selection['adapter'])
        extra = ['--split', 'test', '--selection-file', str(run / 'selection.json')]
    else:
        raise ValueError(name)
    return [str(NATIVE), str(WORKER), 'evaluate', '--output', str(output),
            '--adapter', str(adapter), '--max-new-tokens', '1024', '--resume-results', *extra, *shared]


def verified_predictions(directory, split):
    status = json.loads((directory / 'status.json').read_text())
    assert status['status'] == 'complete'
    labels = read_rows(DATA / f'{split}.labels.jsonl')
    inputs = read_rows(DATA / f'{split}.inputs.jsonl')
    predictions = read_rows(directory / 'predictions.jsonl')
    assert len(predictions) == len(labels)
    import hashlib
    for i, (row, label, inp) in enumerate(zip(predictions, labels, inputs, strict=True), 1):
        expected = label['answer'].lower()
        prompt = '\n'.join(c['text'] for c in inp['messages'][0]['content'] if c['type'] == 'text')
        assert row['sample_id'] == i and row['source_sample_id'] == label['sample_id']
        assert row['expected'] == expected
        assert row['predicted'] == parse_verdict(row['full_response'])
        assert row['correct'] == (row['predicted'] == expected)
        assert row['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['split'] == split and manifest['data_sha256'] == digest(DATA / f'{split}.jsonl')
    assert manifest['dtype'] == 'bfloat16' and manifest['quantization'] is None
    assert manifest['do_sample'] is False and manifest['max_new_tokens'] == 1024
    assert digest(Path(manifest['adapter']) / 'adapter_model.safetensors') == manifest['adapter_sha256']
    return predictions, manifest


def choose_best(entries):
    """No test metrics accepted: maximize validation correct count, earliest tie."""
    if len(entries) != EPOCHS or {r['epoch'] for r in entries} != set(range(1, EPOCHS + 1)):
        raise ValueError('All fifteen distinct epochs must finish validation before selection')
    if any(r.get('split') != 'val' or r.get('total') != 71 for r in entries):
        raise ValueError('Selection only accepts complete 71-row validation results')
    if any(not 0 <= r['correct'] <= 71 for r in entries):
        raise ValueError('Invalid validation correct count')
    return min(entries, key=lambda r: (-r['correct'], r['epoch']))


def validation_report(run):
    entries = []
    for epoch in range(1, EPOCHS + 1):
        directory = run / 'validation' / f'epoch-{epoch}'
        status = directory / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'complete':
            continue
        records, manifest = verified_predictions(directory, 'val')
        assert Path(manifest['adapter']).resolve() == (run / 'train' / f'checkpoint-{epoch * STEPS_PER_EPOCH}').resolve()
        entries.append({**summarize(records), 'split': 'val', 'epoch': epoch, 'step': epoch * STEPS_PER_EPOCH,
                        'adapter': manifest['adapter'], 'adapter_sha256': manifest['adapter_sha256'],
                        'predictions_sha256': digest(directory / 'predictions.jsonl')})
    save_json(run / 'validation_results.json', {'results': entries, 'complete': len(entries) == EPOCHS})
    with (run / 'validation_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'step', 'total', 'correct', 'accuracy',
            'success_recognition_rate', 'failure_recognition_rate', 'invalid_outputs'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(entries)
    return entries


def seal_selection(run, entries):
    winner = choose_best(entries)
    selection = {
        'selection_split': 'val', 'metric': 'final_yes_no_accuracy',
        'tie_break': 'earliest_epoch', 'validation_epochs': list(range(1, EPOCHS + 1)),
        'test_used_for_selection': False, 'epoch': winner['epoch'], 'step': winner['step'],
        'validation_correct': winner['correct'], 'validation_total': 71,
        'validation_accuracy': winner['accuracy'], 'adapter': winner['adapter'],
        'adapter_sha256': winner['adapter_sha256'],
        'validation_prediction_hashes': {str(r['epoch']): r['predictions_sha256'] for r in entries},
        'dataset_manifest_sha256': digest(DATA / 'DATASET_MANIFEST.json'),
    }
    path = run / 'selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == selection, 'Sealed selection changed'
    else:
        assert not (run / 'test').exists(), 'Test must not start before selection'
        save_json(path, selection)
    return selection


def prepare(run):
    if (run / 'plan.json').exists():
        return verify_plan(run)
    validate_dataset(DATA, images=True)
    assert shutil.disk_usage(run.parent).free > 30 * 1024**3
    verification = json.loads((BASE.parent / 'verified.json').read_text())
    for item in verification['files']:
        assert digest(BASE / item['path']) == item['sha256'], item['path']
    assert not json.loads((BASE / 'config.json').read_text()).get('quantization_config')
    scripts = [Path(__file__).resolve(), WORKER, ROOT / 'scripts/qreflector_sft_bf16.py',
               ROOT / 'scripts/eval_gui_reflection.py', ROOT / 'scripts/package_qreflector_v9_prompt.py',
               ROOT / 'scripts/package_adjudicated_condition_data.py']
    files = scripts + [DATA / 'DATASET_MANIFEST.json', RUNTIME, BASE / 'config.json',
                       BASE / 'model.safetensors.index.json', BASE.parent / 'verified.json']
    files += [PROCESSOR_BASE / n for n in ['tokenizer.json', 'tokenizer_config.json',
               'chat_template.jinja', 'preprocessor_config.json', 'generation_config.json']]
    plan = {
        'created_utc': now(), 'dataset': str(DATA), 'train_rows': 880, 'validation_rows': 71, 'test_rows': 1082,
        'initialization': 'original Qwen3-VL-8B-Instruct BF16 base; fresh LoRA; no previous adapter',
        'gpu_index': 7, 'gpu_uuid': GPU_UUID, 'torch_memory_cap_gib': 32,
        'minimum_start_free_gib': 44, 'minimum_runtime_free_gib': 12,
        'training': {'epochs': EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH, 'total_steps': EPOCHS * STEPS_PER_EPOCH,
            'batch_size': 1, 'gradient_accumulation_steps': 8, 'learning_rate': 2e-4,
            'scheduler': 'cosine', 'warmup_ratio': 0.1, 'rank': 16, 'alpha': 16,
            'optimizer': 'adamw_8bit', 'base_weights': 'unquantized BF16',
            'assistant_only_loss': True, 'save_strategy': 'epoch', 'save_total_limit': None,
            'seed': 42, 'data_seed': 42},
        'evaluation': {'validation_epochs': list(range(1, EPOCHS + 1)), 'metric': 'final_yes_no_accuracy',
            'invalid_verdict_counts_as_wrong': True, 'tie_break': 'earliest_epoch',
            'test_policy': 'one full evaluation of the validation winner only',
            'expected_generated_predictions': EPOCHS * 71 + 1082,
            'image_max_side': 512, 'max_new_tokens': 1024, 'greedy': True,
            'pre_post_reason_scored': False},
        'runtime_environment': json.loads(RUNTIME.read_text()),
        'prompt_sha256': digest(DATA / 'prompt_template.txt'),
        'assistant_targets_changed_from_v8': False,
        'prompt_development_used_previous_test_error_analysis': True,
        'pinned_files': {str(p): digest(p) for p in files},
        'base_weights_verified': True,
        'policy': 'Only spawn/wait for own children. No signals to existing processes, GPU resets or service changes.',
    }
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / 'plan.json', plan)
    save_json(run / 'status.json', {'status': 'prepared', 'created_utc': now()})
    return plan


def verify_plan(run):
    plan = json.loads((run / 'plan.json').read_text())
    for filename, expected in plan['pinned_files'].items():
        assert digest(Path(filename)) == expected, f'Pinned file changed: {filename}'
    validate_dataset(DATA)
    return plan


def run_stage(run, name, directory, argv, plan, completed):
    verify_plan(run)
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    expected_status = 'smoke_complete' if name == 'smoke' else 'complete'
    if status.get('status') == expected_status:
        return
    if name == 'train' and directory.exists():
        checkpoints = sorted(directory.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        checkpoints = [p for p in checkpoints if all((p / n).is_file() for n in
                       ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors'])]
        if checkpoints:
            argv += ['--resume', str(checkpoints[-1])]
    while True:
        free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
        if free >= plan['minimum_start_free_gib']:
            break
        save_json(run / 'status.json', {'status': 'waiting_for_gpu_headroom', 'stage': name,
            'free_gib': free, 'required_free_gib': plan['minimum_start_free_gib'],
            'runner_pid': os.getpid(), 'completed_stages': completed, 'updated_utc': now()})
        time.sleep(30)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': GPU_UUID, 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    if name in ('smoke', 'train'):
        env.update(plan['runtime_environment'])
    (run / 'logs').mkdir(exist_ok=True)
    with (run / 'logs' / f'{name}.log').open('a') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        save_json(run / 'status.json', {'status': 'running', 'stage': name,
            'runner_pid': os.getpid(), 'child_pid': child.pid, 'completed_stages': completed,
            'stage_started_utc': now()})
        print(f'{now()} starting {name}, pid={child.pid}', flush=True)
        code = child.wait()
    if code:
        save_json(run / 'status.json', {'status': 'failed', 'stage': name, 'returncode': code,
            'completed_stages': completed, 'log': str(run / 'logs' / f'{name}.log'), 'updated_utc': now()})
        raise RuntimeError(f'{name} exited with {code}; only this job stopped')
    status = json.loads(status_path.read_text())
    assert status['status'] == expected_status, f'Missing completion marker: {name}'


def execute(run, plan):
    (run / 'logs').mkdir(exist_ok=True)
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    completed = []
    for name in ('smoke', 'train'):
        run_stage(run, name, run / name, command(run, name), plan, completed)
        completed.append(name)
    checkpoints = json.loads((run / 'train' / 'checkpoints.json').read_text())
    assert len(checkpoints) == EPOCHS and [c['epoch'] for c in checkpoints] == list(range(1, EPOCHS + 1))
    for epoch in range(1, EPOCHS + 1):
        name = f'validation-epoch-{epoch}'
        run_stage(run, name, run / 'validation' / f'epoch-{epoch}',
                  command(run, 'validation', epoch=epoch), plan, completed)
        completed.append(name)
        validation_report(run)
    selection = seal_selection(run, validation_report(run))
    run_stage(run, 'test', run / 'test', command(run, 'test'), plan, completed)
    records, test_manifest = verified_predictions(run / 'test', 'test')
    assert test_manifest['adapter_sha256'] == selection['adapter_sha256']
    completed.append('test')
    save_json(run / 'results.json', {'selection': selection, 'validation': validation_report(run),
                                   'test': summarize(records), 'completed_utc': now()})
    save_json(run / 'status.json', {'status': 'complete', 'completed_stages': completed,
        'selected_epoch': selection['epoch'], 'test_accuracy': summarize(records)['accuracy'],
        'completed_utc': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert ROOT / 'artifacts/training' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(run)
        if args.prepare_only:
            print(json.dumps({'prepared': str(run), 'epochs': EPOCHS, 'steps': EPOCHS * STEPS_PER_EPOCH,
                              'validation_predictions': EPOCHS * 71, 'test_predictions': 1082}))
        else:
            try:
                execute(run, plan)
            except Exception as error:
                current = json.loads((run / 'status.json').read_text())
                save_json(run / 'status.json', {**current, 'status': 'failed',
                          'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
                raise


if __name__ == '__main__':
    main()
