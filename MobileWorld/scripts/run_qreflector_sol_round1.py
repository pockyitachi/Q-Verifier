#!/usr/bin/env python3
"""Supervise one authorized correction/training/test round, with no validation."""
import argparse
import fcntl
import json
import os
import pwd
import subprocess
import time
from pathlib import Path

import qreflector_sol_round1 as protocol
from qreflector_sol_round1 import ROOT, GPU, all_corrections, load_plan, shared, status

PYTHON = ROOT / 'artifacts/envs/qreflector-native/bin/python'
CPU = ROOT / 'scripts/qreflector_sol_round1.py'
GPU_WORKER = ROOT / 'scripts/qreflector_sol_round1_gpu.py'


class ExistingTeacher:
    """Read-only monitor of this experiment's already running API worker.

    It is never signaled, suspended, reconfigured, or restarted. It consumes the
    original input stream and exits normally after all 880 rows are corrected.
    """
    def __init__(self, run, pid):
        self.run,self.pid,self.returncode=run,pid,None
        proc=Path(f'/proc/{pid}')
        assert pwd.getpwuid(proc.stat().st_uid).pw_name=='linqiang'
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        assert str(CPU).encode() in argv and b'teachers' in argv and str(run).encode() in argv
        self.start_ticks=(proc/'stat').read_text().split()[21]

    def poll(self):
        proc=Path(f'/proc/{self.pid}/stat')
        try:
            state=proc.read_text().split()
            alive=state[21]==self.start_ticks and state[2]!='Z'
        except FileNotFoundError:
            alive=False
        if alive:
            return None
        self.returncode=0 if completed(self.run,'teachers') else 1
        return self.returncode

    def wait(self):
        while self.poll() is None:
            time.sleep(3)
        return self.returncode


def free_gib():
    return float(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())/1024


def completed(run, stage):
    p=run/f'{stage}_status.json'
    return p.exists() and shared.read_json(p).get('status')=='complete'


def spawn(run, stage, gpu=False):
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=GPU if gpu else '')
    log=(run/'logs'/f'{stage}.log').open('a')
    child=subprocess.Popen([str(PYTHON),'-u',str(GPU_WORKER if gpu else CPU),stage,
        '--run-dir',str(run)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT)
    log.close()
    status(run,'pipeline',status='running',stage=stage,runner_pid=os.getpid(),child_pid=child.pid)
    print(f'Started {stage}: PID {child.pid}',flush=True)
    return child


def check_teacher(run, api):
    if api is not None and api.poll() is not None and (api.returncode or not completed(run,'teachers')):
        raise RuntimeError('Sol correction process failed; inspect teachers.log')


def run_stage(run, stage, gpu=False, api=None):
    load_plan(run)
    if completed(run,stage):
        return
    retries=0
    while True:
        if gpu:
            while free_gib()<protocol.CAP_GIB+protocol.RESERVE_GIB:
                check_teacher(run,api)
                status(run,'pipeline',status='waiting_for_gpu_headroom',stage=stage,
                       free_gib=free_gib(),required_free_gib=protocol.CAP_GIB+protocol.RESERVE_GIB,
                       runner_pid=os.getpid())
                time.sleep(30)
        child=spawn(run,stage,gpu)
        # Own GPU child is allowed to finish/resume even if the API child fails.
        # No signal is ever sent to any pre-existing GPU process.
        code=child.wait()
        if code==75 and gpu and retries<3:
            retries+=1
            status(run,'pipeline',status='waiting_for_gpu_headroom',stage=stage,retries=retries)
            time.sleep(30)
            continue
        if code or not completed(run,stage):
            raise RuntimeError(f'{stage} failed (code {code}); inspect its own log')
        check_teacher(run,api)
        return


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--existing-teacher-pid',type=int)
    args=p.parse_args();run=args.run_dir.resolve()
    load_plan(run)
    (run/'logs').mkdir(exist_ok=True)
    with (run/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        api=None
        try:
            if not completed(run,'teachers'):
                api=ExistingTeacher(run,args.existing_teacher_pid) if args.existing_teacher_pid else spawn(run,'teachers')
            # Correct the already available initial-student pilot drafts first.
            while not completed(run,'smoke') and sum(r['accepted'] for r in all_corrections(run))<8:
                check_teacher(run,api)
                status(run,'pipeline',status='waiting_for_smoke_targets',runner_pid=os.getpid(),
                       corrections_completed=len(all_corrections(run)))
                time.sleep(3)
            run_stage(run,'smoke',gpu=True,api=api)
            assert shared.read_json(run/'smoke_status.json')['adapter_changed']
            run_stage(run,'generate',gpu=True,api=api)
            if api is not None:
                status(run,'pipeline',status='running',stage='finish_teacher_corrections',runner_pid=os.getpid(),child_pid=api.pid)
                code=api.wait()
                assert code==0 and completed(run,'teachers')
            run_stage(run,'freeze')
            run_stage(run,'train',gpu=True)
            run_stage(run,'test',gpu=True)
            run_stage(run,'report')
            status(run,'pipeline',status='complete',runner_pid=os.getpid(),
                   completed_stages=['teachers','smoke','generate','freeze','train','test','report'],validation_used=False)
        except BaseException as error:
            status(run,'pipeline',status='failed',runner_pid=os.getpid(),error_type=type(error).__name__,error=str(error))
            # Do not signal any process, including a still-running API worker.
            # Completed responses and stage state remain available for resuming.
            raise


if __name__=='__main__':
    main()
