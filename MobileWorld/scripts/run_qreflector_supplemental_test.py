#!/usr/bin/env python3
"""Queue an explicitly requested supplemental test after the primary run exits.

Reuse the frozen v8 inference implementation. A process-local authorization
hook replaces its validation-winner gate only in this supplemental worker.
It checks the separately pinned user-requested checkpoint instead. No original
script, selection record, result, GPU setting, or existing process is changed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import qreflector_sft_v8 as worker
from eval_gui_reflection import save_json, summarize
from run_qreflector_v8 import NATIVE, verified_predictions, verify_plan


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessor_ready(run):
    """Require the full primary pipeline to finish and release its run lock."""
    for path in (run / 'status.json', run / 'test/status.json'):
        if not path.exists() or json.loads(path.read_text()).get('status') != 'complete':
            return False
    with (run / 'runner.lock').open('r') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return True


def verify_supplement(output):
    plan = json.loads((output / 'supplemental_plan.json').read_text())
    assert plan['purpose'] == 'user_requested_supplemental_test'
    assert plan['epoch'] == 7 and plan['step'] == 770
    assert plan['test_rows'] == 1082 and plan['gpu_uuid'] == worker.GPU_UUID
    assert plan['primary_selection_replaced'] is False
    for filename, expected in plan['pinned_files'].items():
        assert worker.digest(Path(filename)) == expected, f'Pinned supplemental input changed: {filename}'
    verify_plan(Path(plan['primary_run']))
    return plan


def prepare(run, output):
    if (output / 'supplemental_plan.json').exists():
        return verify_supplement(output)
    original = verify_plan(run)
    selected = json.loads((run / 'selection.json').read_text())
    assert selected['epoch'] == 2 and selected['step'] == 220
    adapter = run / 'train/checkpoint-770'
    state = json.loads((adapter / 'trainer_state.json').read_text())
    assert state['epoch'] == 7.0 and state['global_step'] == 770
    files = [Path(__file__).resolve(), run / 'plan.json', run / 'selection.json',
             adapter / 'adapter_model.safetensors', adapter / 'adapter_config.json',
             adapter / 'trainer_state.json', worker.DATA / 'DATASET_MANIFEST.json',
             worker.DATA / 'test.jsonl', worker.DATA / 'test.inputs.jsonl',
             worker.DATA / 'prompt_template.txt']
    plan = {
        'created_utc': now(), 'purpose': 'user_requested_supplemental_test',
        'user_request': 'epoch2跑完在跑下epoch7',
        'primary_run': str(run), 'epoch': 7, 'step': 770,
        'adapter': str(adapter), 'adapter_sha256': worker.digest(adapter / 'adapter_model.safetensors'),
        'dataset': str(worker.DATA), 'test_rows': 1082,
        'gpu_uuid': worker.GPU_UUID, 'gpu_index': 7,
        'memory_cap_gib': 32, 'minimum_start_free_gib': 44,
        'image_max_side': original['evaluation']['image_max_side'],
        'max_new_tokens': original['evaluation']['max_new_tokens'],
        'dtype': 'bfloat16', 'quantization': None, 'do_sample': False,
        'metric': 'final_yes_no_accuracy', 'primary_selection_replaced': False,
        'primary_selected_epoch': 2,
        'interpretation': 'Additional requested comparison after seeing partial epoch-2 test results; not validation-selected.',
        'wait_policy': 'Require primary pipeline completion, released lock, verified full test and GPU headroom.',
        'pinned_files': {str(p): worker.digest(p) for p in files},
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'supplemental_plan.json', plan)
    save_json(output / 'queue_status.json', {'status': 'prepared', 'updated_utc': now()})
    return plan


def supplemental_gate(args):
    plan = verify_supplement(args.output)
    assert args.selection_file.resolve() == (args.output / 'supplemental_plan.json').resolve()
    assert args.split == 'test' and args.limit == 1082
    assert args.adapter.resolve() == Path(plan['adapter']).resolve()
    assert worker.digest(args.adapter / 'adapter_model.safetensors') == plan['adapter_sha256']
    assert args.data.resolve() == Path(plan['dataset']).resolve()
    assert args.gpu_uuid == plan['gpu_uuid']
    assert args.memory_cap_gib == plan['memory_cap_gib']
    assert args.image_max_side == plan['image_max_side']
    assert args.max_new_tokens == plan['max_new_tokens']
    run = Path(plan['primary_run'])
    assert predecessor_ready(run), 'Primary evaluation must finish first'
    _, manifest = verified_predictions(run / 'test', 'test')
    primary = json.loads((run / 'selection.json').read_text())
    assert manifest['adapter_sha256'] == primary['adapter_sha256']


def evaluate_supplement(output):
    plan = verify_supplement(output)
    args = Namespace(data=Path(plan['dataset']), output=output, adapter=Path(plan['adapter']),
                     split='test', limit=1082, gpu_uuid=plan['gpu_uuid'],
                     memory_cap_gib=plan['memory_cap_gib'], image_max_side=plan['image_max_side'],
                     max_new_tokens=plan['max_new_tokens'], resume_results=True,
                     selection_file=output / 'supplemental_plan.json')
    # This affects only this new Python process. The original process and files
    # retain their validation-only gate. All inference and scoring code is reused.
    original_gate = worker.validate_test_selection
    try:
        worker.validate_test_selection = supplemental_gate
        worker.evaluate(args)
    finally:
        worker.validate_test_selection = original_gate


def queue(output):
    plan = verify_supplement(output)
    run = Path(plan['primary_run'])
    status_path = output / 'queue_status.json'
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads(status_path.read_text()).get('status') == 'complete':
            return
        try:
            while not predecessor_ready(run):
                previous = json.loads((run / 'status.json').read_text())
                progress = json.loads((run / 'test/status.json').read_text())
                save_json(status_path, {'status': 'waiting_for_epoch2_completion',
                    'queue_pid': os.getpid(), 'predecessor_status': previous['status'],
                    'predecessor_completed': progress.get('completed'),
                    'predecessor_total': 1082, 'updated_utc': now()})
                time.sleep(30)
            verify_supplement(output)
            primary_rows, _ = verified_predictions(run / 'test', 'test')
            while True:
                free = float(subprocess.check_output(['nvidia-smi', '-i', plan['gpu_uuid'],
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
                if free >= plan['minimum_start_free_gib']:
                    break
                save_json(status_path, {'status': 'waiting_for_gpu_headroom', 'queue_pid': os.getpid(),
                    'free_gib': free, 'required_free_gib': plan['minimum_start_free_gib'], 'updated_utc': now()})
                time.sleep(30)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': plan['gpu_uuid'], 'PYTHONUNBUFFERED': '1',
                   'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
            with (output / 'evaluation.log').open('a') as log:
                child = subprocess.Popen([str(NATIVE), str(Path(__file__).resolve()),
                    '--run-dir', str(run), '--output', str(output), '--worker'],
                    cwd=worker.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                save_json(status_path, {'status': 'running', 'queue_pid': os.getpid(),
                    'child_pid': child.pid, 'epoch': 7, 'started_utc': now()})
                code = child.wait()
            if code:
                raise RuntimeError(f'Supplemental test child exited with {code}')
            supplemental_rows, manifest = verified_predictions(output, 'test')
            assert manifest['adapter_sha256'] == plan['adapter_sha256']
            assert all(a['prompt_sha256'] == b['prompt_sha256'] and a['input_tokens'] == b['input_tokens']
                       for a, b in zip(primary_rows, supplemental_rows, strict=True))
            verify_supplement(output)
            save_json(output / 'comparison.json', {
                'primary_validation_selected_epoch': 2,
                'primary': summarize(primary_rows), 'supplemental_epoch': 7,
                'supplemental': summarize(supplemental_rows),
                'primary_selection_replaced': False, 'completed_utc': now()})
            save_json(status_path, {'status': 'complete', 'epoch': 7, 'completed_utc': now()})
        except Exception as error:
            save_json(status_path, {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                   'updated_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    assert worker.ROOT / 'artifacts/training' in run.parents
    assert output == run / 'supplemental_epoch7_test'
    if args.worker:
        evaluate_supplement(output)
    else:
        plan = prepare(run, output)
        if args.prepare_only:
            print(json.dumps({'prepared': str(output), 'epoch': plan['epoch'], 'test_rows': 1082}))
        else:
            queue(output)


if __name__ == '__main__':
    main()
