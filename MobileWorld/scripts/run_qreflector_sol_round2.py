#!/usr/bin/env python3
"""Run geometry ablation, fresh Sol corrections, one training epoch and test.

No signals, service changes, resets, or operations on existing processes.
"""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path

import qreflector_sol_round2 as protocol
from qreflector_sol_round2 import ROOT, GPU, load_plan, shared, status

PYTHON = ROOT / 'artifacts/envs/qreflector-native/bin/python'
CPU = ROOT / 'scripts/qreflector_sol_round2.py'
GPU_WORKER = ROOT / 'scripts/qreflector_sol_round2_gpu.py'


def completed(run, stage):
    path = run / f'{stage}_status.json'
    return path.exists() and shared.read_json(path).get('status') == 'complete'


def availability():
    free = float(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'], text=True).strip()) / 1024
    occupied = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid',
                 '--format=csv,noheader,nounits'], text=True).strip()
    return free, not occupied


def spawn(run, stage, gpu=False):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=GPU if gpu else '', USE_TF='0', USE_FLAX='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with (run / 'logs' / f'{stage}.log').open('a') as log:
        child = subprocess.Popen([str(PYTHON), '-u', str(GPU_WORKER if gpu else CPU), stage,
            '--run-dir', str(run)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    status(run, 'pipeline', status='running', stage=stage, runner_pid=os.getpid(), child_pid=child.pid)
    print(f'Started {stage}: PID {child.pid}', flush=True)
    return child


def run_stage(run, stage, gpu=False):
    load_plan(run)
    if completed(run, stage):
        return
    for attempt in range(4):
        if gpu:
            while True:
                free, empty = availability()
                if empty and free >= protocol.CAP_GIB + protocol.RESERVE_GIB:
                    break
                status(run, 'pipeline', status='waiting_for_empty_gpu', stage=stage,
                       free_gib=free, gpu_empty=empty, runner_pid=os.getpid())
                time.sleep(30)
        child = spawn(run, stage, gpu)
        code = child.wait()
        if code == 75 and gpu and attempt < 3:
            time.sleep(30)
            continue
        if code or not completed(run, stage):
            raise RuntimeError(f'{stage} failed (code {code}); inspect its own log')
        return


def finish_teachers(run, api):
    """Resume cached API work at most twice after a failed process, never signal it."""
    for attempt in range(3):
        code = api.wait()
        if code == 0 and completed(run, 'teachers'):
            return
        if attempt == 2:
            raise RuntimeError('Teacher stage failed after bounded process retries')
        status(run, 'pipeline', status='retrying_teacher_stage', attempt=attempt + 1,
               runner_pid=os.getpid())
        time.sleep(30)
        api = spawn(run, 'teachers')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    plan, _ = load_plan(run)
    (run / 'logs').mkdir(exist_ok=True)
    # Separate pipeline invocations cannot overlap in this experiment directory.
    with (run / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Complete the unchanged-weight input comparison first.
            run_stage(run, 'test_before', gpu=True)
            api = None if completed(run, 'teachers') else spawn(run, 'teachers')
            run_stage(run, 'generate', gpu=True)
            if api is not None:
                status(run, 'pipeline', status='running', stage='finish_teacher_corrections',
                       runner_pid=os.getpid(), child_pid=api.pid)
                finish_teachers(run, api)
            run_stage(run, 'freeze')
            run_stage(run, 'smoke', gpu=True)
            assert shared.read_json(run / 'smoke_status.json')['adapter_changed']
            run_stage(run, 'train', gpu=True)
            run_stage(run, 'test', gpu=True)
            run_stage(run, 'report')
            status(run, 'pipeline', status='complete', runner_pid=os.getpid(),
                   completed_stages=plan['stage_order'], validation_used=False,
                   report=str(run / 'report.txt'), final_adapter=str(run / 'train/final'))
        except BaseException as error:
            status(run, 'pipeline', status='failed', runner_pid=os.getpid(),
                   error_type=type(error).__name__, error=str(error))
            # Even on failure, do not send signals to any process.
            raise


if __name__ == '__main__':
    main()
