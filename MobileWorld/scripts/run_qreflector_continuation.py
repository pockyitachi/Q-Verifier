#!/usr/bin/env python3
"""Run BF16 weight continuation and three validation evaluations, serially."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from qreflector_continue_bf16 import DATA, GPU_UUID, ROOT, SOURCE, digest, read_rows, validate_dataset
from eval_gui_reflection import gpu_snapshot, save_json, summarize


def stages(run):
    worker = ROOT / 'scripts/qreflector_continue_bf16.py'
    common = ['--gpu-uuid', GPU_UUID, '--memory-cap-gib', '40']
    result = [('train', [str(ROOT / 'artifacts/envs/qreflector-sft/bin/python'),
                        str(worker), 'train', '--output', str(run / 'training'), *common])]
    for cumulative, adapter in [(333, SOURCE), (444, run / 'training/checkpoint-111'),
                                (555, run / 'training/checkpoint-222')]:
        result.append((f'val-{cumulative}', [str(ROOT / 'artifacts/envs/qreflector-native/bin/python'),
            str(worker), 'evaluate', '--output', str(run / f'val-{cumulative}'),
            '--adapter', str(adapter), '--limit', '71', '--resume-results', *common]))
    return result


def report(run):
    targets = read_rows(DATA / 'val.jsonl')
    summaries = []
    for step in (333, 444, 555):
        rows = read_rows(run / f'val-{step}/predictions.jsonl')
        assert len(rows) == 71
        for i, (row, target) in enumerate(zip(rows, targets), 1):
            expected = json.loads(target['messages'][1]['content'][0]['text'])['answer'].lower()
            prompt = '\n'.join(c['text'] for c in target['messages'][0]['content'] if c['type'] == 'text')
            assert row['sample_id'] == i and row['expected'] == expected
            assert row['correct'] == (row['predicted'] == expected)
            assert row['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
        summaries.append(dict(summarize(rows), cumulative_step=step))
    save_json(run / 'validation_results.json', {
        'results': summaries, 'split': 'val', 'score': 'Final Yes/No only; invalid is incorrect',
        'test_set_evaluated': False, 'exact_resume': False,
        'continuation': 'BF16 step-333 adapter + 2 epochs, new AdamW8bit optimizer, 2e-5 cosine, 5% warmup',
    })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert ROOT / 'artifacts/training' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_dataset(DATA, images=True)
    original = SOURCE.parents[1]
    old_plan = json.loads((original / 'plan.json').read_text())
    for path, sha in old_plan['scripts_sha256'].items():
        assert digest(Path(path)) == sha, 'Original experiment script changed'
    assert json.loads((SOURCE / 'trainer_state.json').read_text())['global_step'] == 333
    runtime = json.loads((original / 'runtime_environment.json').read_text())
    plan = {
        'source': str(SOURCE), 'source_sha256': digest(SOURCE / 'adapter_model.safetensors'),
        'dataset': str(DATA), 'frozen_sha256': digest(DATA / 'FROZEN.json'),
        'gpu_uuid': GPU_UUID, 'memory_cap_gib': 40, 'minimum_startup_free_gib': 48,
        'additional_epochs': 2, 'additional_steps': 222, 'cumulative_final_step': 555,
        'learning_rate': 2e-5, 'scheduler': 'cosine', 'warmup_ratio': 0.05,
        'new_optimizer': True, 'exact_resume': False, 'data_seed': 45,
        'evaluation_split': 'val', 'evaluation_steps': [333, 444, 555],
        'scripts_sha256': {str(p): digest(p) for p in [Path(__file__), ROOT / 'scripts/qreflector_continue_bf16.py']},
        'stages': [{'name': name, 'argv': argv} for name, argv in stages(run)],
        'policy': 'Only start and wait for own children; never signal existing processes or modify GPU/services',
    }
    plan_path = run / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        save_json(plan_path, plan)
    if args.prepare_only:
        print(json.dumps(plan, indent=2), flush=True)
        return
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    (run / 'logs').mkdir(exist_ok=True)
    completed = []
    for name, argv in stages(run):
        output = run / ('training' if name == 'train' else name)
        status_path = output / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') == 'complete':
            if name == 'train':
                assert status['cumulative_step'] == 555
            else:
                assert len(read_rows(output / 'predictions.jsonl')) == 71
            completed.append(name)
            continue
        if name == 'train' and status_path.exists():
            raise RuntimeError('Partial training requires explicit recovery; refusing to overwrite')
        while True:
            free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
            if free >= 48:
                break
            save_json(run / 'status.json', {'status': 'waiting_for_gpu_headroom', 'stage': name,
                'free_gib': free, 'completed_stages': completed})
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
    report(run)
    assert digest(SOURCE / 'adapter_model.safetensors') == plan['source_sha256']
    save_json(run / 'status.json', {'status': 'complete', 'completed_stages': completed,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
