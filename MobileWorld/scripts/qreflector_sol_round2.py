#!/usr/bin/env python3
"""Second Sol correction round with shared dynamic geometry; no validation selection.

All drafts come from the fixed initial student. Teacher correction completes
before one pass of assistant-only CE training. The initial and final adapters
are each tested once with identical geometry-augmented inputs.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import http.client
import json
import math
import os
import random
import re
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path

import qreflector_luna_correction as correction
import qreflector_opsd as shared
import qreflector_teacher_pilot_v9 as pilot
import qreflector_geometry_prompt as geometry_prompt
from qreflector_sft_bf16 import (
    BASE, PROCESSOR_BASE, check_bf16_base, environment, gpu_setup, read_rows, strict_structure,
)
from eval_gui_reflection import parse_verdict, summarize

ROOT, DATA = shared.ROOT, pilot.DATA
ROUND1 = ROOT / 'artifacts/training/qreflector_sol_round1_v9_epoch9_20261008_gpu7'
ADAPTER = ROUND1 / 'train/final'
GPU = 'GPU-fb63c19e-9ce2-10d1-3bfe-302cd3125430'
BASELINE = ROUND1 / 'test/predictions.jsonl'
SEED = 20261008
WORKERS = 4
CAP_GIB, RESERVE_GIB = 32, 24
HOLD_IDS = {'train-90': 'Unresolved task-progress annotation: X Chat passcode versus photo/link-sharing goal.'}
RAW_TEACHER_REQUEST = correction.teacher_request
TEACHER_INSTRUCTIONS = pilot.TEACHER_INSTRUCTIONS + '''
The supplied image geometry is authoritative metadata, not an action-result
annotation. Action coordinates are pixels in the original BEFORE screenshot,
with origin at top left, x increasing rightward and y increasing downward.
Use the provided normalized or resized coordinates to locate targets in the
resized image. Do not compare original pixel coordinates directly with resized
image dimensions. Do not infer a tap target only from AFTER or blindly follow
an inconsistent action description. If a target is near a boundary or unclear,
state that uncertainty rather than claiming a precise hit.
Switch_App with no target argument invokes Android's recent-apps/Overview view;
selecting a specific app card may be a later action. Opening that view can be
valid intermediate progress. A scroll used to inspect more of a relevant page
need not immediately expose the final sought attribute. These conventions do
not make unrelated actions, incorrect text values, or wrong targets successful.
'''


def rows_if_exists(path):
    """Read complete records while another process appends a single JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n' if '\n' in text else ''
    return [json.loads(line) for line in text.splitlines() if line]


def context_of(row):
    text = next(c['text'] for c in row['messages'][0]['content'] if c['type'] == 'text')
    return json.loads(text.split('=== Context ===')[1].split('=== Image Reference ===')[0])


def teacher_request(row, draft, config):
    # The same geometry-augmented user prefix is seen by both teacher and student.
    augmented = geometry_prompt.augment_row(row, Path(config['data']), config['image_max_side'])
    payload = RAW_TEACHER_REQUEST(augmented, draft, config)
    payload['instructions'] = TEACHER_INSTRUCTIONS
    return payload


def eligible(record):
    if record['sample_id'] in HOLD_IDS:
        return False, 'predeclared_annotation_hold'
    valid, why = correction.validate_correction(record.get('parsed'), record['expected'])
    if record.get('status') != 'completed':
        return False, 'teacher_response_incomplete'
    if not valid:
        return valid, why
    reason = record['parsed']['corrected_response']['reason']
    if not all(marker in reason for marker in ('Precondition check:', 'Postcondition check:', 'Task progress:')):
        return False, 'missing_reason_checks'
    text = json.dumps(record['parsed']['corrected_response'])
    if re.search(r'\b(reference annotation|student draft|teacher response)\b', text, re.I):
        return False, 'privileged_context_mentioned'
    return True, 'accepted'


def status(run, status_name, **values):
    shared.write_json(run / f'{status_name}_status.json', {'pid': os.getpid(), 'updated_unix': time.time(), **values})


def prepare(run):
    assert not run.exists(), 'New run directory required'
    manifest = shared.read_json(DATA / 'DATASET_MANIFEST.json')
    assert manifest['status'] == 'frozen'
    names = ['train.jsonl', 'train.inputs.jsonl', 'train.labels.jsonl', 'train.line_map.jsonl',
             'train.input_manifest.jsonl', 'test.jsonl', 'test.inputs.jsonl', 'test.labels.jsonl',
             'prompt_template.txt', 'quarantine.jsonl']
    for name in names:
        assert shared.digest(DATA / name) == manifest['artifact_sha256'][name], name
    rows, labels = read_rows(DATA / 'train.jsonl'), read_rows(DATA / 'train.labels.jsonl')
    inputs = read_rows(DATA / 'train.inputs.jsonl')
    assert len(rows) == len(labels) == len(inputs) == 880
    assert shared.read_json(ROUND1 / 'pipeline_status.json')['status'] == 'complete'
    prior_test = shared.read_json(ROUND1 / 'test/manifest.json')
    assert shared.digest(ADAPTER / 'adapter_model.safetensors') == prior_test['adapter_sha256']
    samples, image_hashes = [], {}
    for i, (row, label, inp) in enumerate(zip(rows, labels, inputs, strict=True)):
        assert row['messages'][0] == inp['messages'][0]
        assert label['answer'] == shared.target_of(row)['answer']
        assert label['sample_id'] not in manifest['quarantined_ids']
        for c in row['messages'][0]['content']:
            if c['type'] == 'image':
                image_hashes[c['image']] = shared.digest(DATA / c['image'])
        augmented = geometry_prompt.augment_row(row, DATA)
        assert context_of(augmented) == context_of(row)
        assert shared.target_of(augmented) == shared.target_of(row)
        samples.append({'sample_id': label['sample_id'], 'train_line': i+1, 'row': augmented,
                        'student_messages_sha256': correction.digest_json(correction.student_messages(augmented)),
                        'expected': label['answer'], 'geometry': geometry_prompt.geometry(row, DATA)})
    # Audit all test inputs without using their outputs for tuning or training.
    test_rows = read_rows(DATA / 'test.jsonl')
    assert len(test_rows) == 1082
    for row in test_rows:
        augmented = geometry_prompt.augment_row(row, DATA)
        assert context_of(augmented) == context_of(row)
        assert augmented['messages'][1] == row['messages'][1]
        for c in row['messages'][0]['content']:
            if c['type'] == 'image':
                image_hashes[c['image']] = shared.digest(DATA / c['image'])
    run.mkdir(parents=True)
    pilot.write_jsonl(run / 'samples.jsonl', samples)
    pilot.write_jsonl(run / 'student_rollouts.jsonl', [])
    pinned = [Path(__file__).resolve(), ROOT / 'scripts/qreflector_sol_round2_gpu.py',
              ROOT / 'scripts/run_qreflector_sol_round2.py', ROOT / 'scripts/qreflector_geometry_prompt.py',
              ROOT / 'scripts/qreflector_luna_correction.py', ROOT / 'scripts/qreflector_opsd.py',
              ROOT / 'scripts/qreflector_teacher_pilot_v9.py', ROOT / 'scripts/qreflector_sft_bf16.py',
              ROOT / 'scripts/eval_gui_reflection.py', DATA / 'DATASET_MANIFEST.json',
              ADAPTER / 'adapter_model.safetensors', ADAPTER / 'adapter_config.json',
              BASE / 'config.json', BASE / 'model.safetensors.index.json',
              ROUND1 / 'plan.json', ROUND1 / 'comparison.json', BASELINE]
    pinned += [DATA / name for name in names]
    pinned += [ADAPTER / name for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                                         'preprocessor_config.json']]
    pinned += [ADAPTER / name for name in ['processor_config.json', 'video_preprocessor_config.json',
                'special_tokens_map.json', 'added_tokens.json', 'vocab.json'] if (ADAPTER / name).exists()]
    prior = shared.read_json(ROUND1 / 'plan.json')
    plan = {'created_unix': time.time(), 'method': 'batched_student_response_correction_distillation',
            'not_token_kl_opd': True, 'round': 2, 'data': str(DATA), 'adapter': str(ADAPTER),
            'base': str(BASE), 'initialization': 'Round-1 final BF16 LoRA student, unchanged source weights',
            'source_adapter_sha256': shared.digest(ADAPTER / 'adapter_model.safetensors'),
            'training_inputs': 880, 'imported_matching_student_drafts': 0, 'new_student_drafts': 880,
            'sampling': {'temperature': 1., 'top_p': 1., 'top_k': 0, 'max_new_tokens': 1024, 'seed': SEED},
            'image_max_side': 512, 'student_prompt_unchanged': False,
            'prompt_change': 'Original v9 + dynamic original dimensions and normalized current-action coordinates',
            'geometry_shared_by_teacher_and_student': True,
            'teacher_model': 'gpt-6.1-sol', 'teacher_reasoning': 'medium',
            'teacher_max_output_tokens': 4096, 'teacher_workers': WORKERS, 'max_api_calls_per_worker': 275,
            'teacher_instructions': TEACHER_INSTRUCTIONS,
            'teacher_privilege': 'Train-only fallible original four-field reference and new round-1-student draft',
            'known_annotation_holds': HOLD_IDS, 'minimum_accepted_fraction': .75,
            'training': prior['training'],
            'evaluation': {'validation': False, 'split': 'test', 'samples': 1082,
                           'checkpoint_policy': 'Predetermined final one-epoch correction checkpoint only',
                           'prompt': 'v9 + shared geometry, identical before and after round 2',
                           'greedy': True, 'max_new_tokens': 1024,
                           'score': 'Final Yes/No; invalid outputs incorrect; pre/post/reason unscored'},
            'stage_order': ['test_before', 'generate_and_teachers', 'freeze', 'smoke', 'train', 'test', 'report'],
            'gpu_index': 2, 'gpu_uuid': GPU, 'gpu_memory_cap_gib': CAP_GIB, 'reserve_free_gib': RESERVE_GIB,
            'minimum_start_free_gib': CAP_GIB + RESERVE_GIB,
            'baseline_predictions': str(BASELINE), 'test_sha256': manifest['artifact_sha256']['test.jsonl'],
            'samples_sha256': shared.digest(run / 'samples.jsonl'),
            'image_sha256': image_hashes, 'pinned_sha256': {str(p): shared.digest(p) for p in pinned},
            'process_policy': 'Only own children; start GPU stages when GPU 2 is empty; no signals, resets or service changes',
            'result_policy': 'Run both predeclared evaluations regardless of first accuracy; no test-selected training choices'}
    shared.write_json(run / 'plan.json', plan)
    shared.write_json(run / 'prompt_example.json', {'student_messages': correction.student_messages(samples[0]['row']),
                                                  'teacher_instructions': TEACHER_INSTRUCTIONS})
    status(run, 'generate', status='prepared', completed=0, total=880)
    status(run, 'pipeline', status='prepared')
    print(json.dumps({'prepared': str(run), 'train': 880, 'new_drafts': 880, 'test_runs': 2, 'validation': False}))


def load_plan(run):
    plan = shared.read_json(run / 'plan.json')
    assert shared.digest(run / 'samples.jsonl') == plan['samples_sha256']
    for path, sha in plan['pinned_sha256'].items():
        assert shared.digest(path) == sha, f'Pinned file changed: {path}'
    return plan, read_rows(run / 'samples.jsonl')


def all_corrections(run):
    records = []
    for worker in range(WORKERS):
        records.extend(rows_if_exists(run / 'teachers' / str(worker) / 'corrections.jsonl'))
    assert len(records) == len({r['sample_id'] for r in records}), 'Duplicate correction IDs'
    return records


def correct_with_transport_retries(client, row, draft):
    """Retry interrupted/invalid HTTP bodies without changing the request.

    The shared client already retries HTTP status errors and network timeouts.
    Every additional attempt here still goes through its cache and API-call cap.
    Do not log partial response bodies, which could contain private data.
    """
    for attempt in range(3):
        try:
            return client.correct(row, draft)
        except (http.client.HTTPException, ConnectionError, json.JSONDecodeError) as error:
            correction.append_json(client.ledger, {
                'event': 'interrupted_response', 'error_type': type(error).__name__,
                'transport_attempt': attempt + 1, 'time_unix': time.time()})
            if attempt == 2:
                raise RuntimeError('Teacher response interrupted after three attempts') from None
            time.sleep(2 ** (attempt + 1))


def teacher_worker(run, plan, samples, worker):
    dest = run / 'teachers' / str(worker)
    dest.mkdir(parents=True, exist_ok=True)
    config = {'data': str(DATA), 'teacher_model': 'gpt-6.1-sol', 'teacher_reasoning': 'medium',
              'teacher_max_output_tokens': 4096, 'max_api_calls': 275, 'image_max_side': 512}
    assigned = {s['sample_id']: s for i,s in enumerate(samples) if i % WORKERS == worker}
    records = rows_if_exists(dest / 'corrections.jsonl')
    done = {r['sample_id'] for r in records}
    assert len(done) == len(records) and done <= set(assigned)
    client = correction.LunaClient(dest, config, ROOT / '.env')
    while len(done) < len(assigned):
        drafts = {r['sample_id']: r for r in rows_if_exists(run / 'student_rollouts.jsonl')}
        available = sorted((set(assigned) - done) & set(drafts), key=lambda sid: assigned[sid]['train_line'])
        if not available:
            generation_state = run / 'generate_status.json'
            if generation_state.exists() and shared.read_json(generation_state).get('status') == 'failed':
                raise RuntimeError('Student generation failed; this API worker exits without signaling any process')
            status(run, f'teacher_{worker}', status='waiting_for_student', completed=len(done), total=len(assigned))
            time.sleep(3)
            continue
        for sid in available:
            sample, draft = assigned[sid], drafts[sid]
            assert draft['student_messages_sha256'] == sample['student_messages_sha256']
            status(run, f'teacher_{worker}', status='correcting', completed=len(done),
                   total=len(assigned), active_sample_id=sid, api_calls=client.calls)
            result, cached = correct_with_transport_retries(client, sample['row'], draft['full_response'])
            assert result['teacher_model'] == 'gpt-6.1-sol', 'Unexpected teacher model returned'
            record = {**result, 'sample_id': sid, 'train_line': sample['train_line'],
                      'expected': sample['expected'], 'draft_sha256': correction.digest_json(draft['full_response']),
                      'student_messages_sha256': sample['student_messages_sha256'], 'cached': cached}
            record['accepted'], record['decision'] = eligible(record)
            correction.append_json(dest / 'corrections.jsonl', record)
            done.add(sid)
            status(run, f'teacher_{worker}', status='running', completed=len(done), total=len(assigned), api_calls=client.calls)
            if len(done) % 20 == 0:
                print(f'Sol worker {worker}: {len(done)}/{len(assigned)}', flush=True)
    status(run, f'teacher_{worker}', status='complete', completed=len(done), total=len(assigned), api_calls=client.calls)


def monitored_teacher_worker(run, plan, samples, worker):
    try:
        return teacher_worker(run, plan, samples, worker)
    except Exception as error:
        status(run, f'teacher_{worker}', status='failed',
               error_type=type(error).__name__, error=str(error))
        raise


def teachers(run):
    plan, samples = load_plan(run)
    correction.teacher_request = teacher_request
    correction.TEACHER_INSTRUCTIONS = TEACHER_INSTRUCTIONS
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(monitored_teacher_worker, run, plan, samples, i) for i in range(WORKERS)]
        for future in futures:
            future.result()
    records = all_corrections(run)
    assert len(records) == 880
    status(run, 'teachers', status='complete', completed=880,
           accepted=sum(r['accepted'] for r in records), decisions=dict(Counter(r['decision'] for r in records)))


def target_sample(sample, record):
    assert sample['sample_id'] == record['sample_id'] and record['accepted']
    target = {k: record['parsed']['corrected_response'][k] for k in shared.FIELDS}
    # Original user prefix is preserved; only the assistant target is replaced.
    row = copy.deepcopy(sample['row'])
    row['messages'][1]['content'][0]['text'] = json.dumps(target, ensure_ascii=False)
    assert row['messages'][0] == sample['row']['messages'][0]
    return {'sample_id': sample['sample_id'], 'row': row, 'teacher_request_sha256': record['request_sha256']}


def freeze(run):
    plan, samples = load_plan(run)
    records = all_corrections(run)
    assert len(records) == 880
    by_id = {r['sample_id']: r for r in records}
    drafts = {r['sample_id']: r for r in rows_if_exists(run / 'student_rollouts.jsonl')}
    accepted, rejected = [], []
    for sample in samples:
        sid = sample['sample_id']
        record = by_id[sid]
        assert record['draft_sha256'] == correction.digest_json(drafts[sid]['full_response'])
        assert record['accepted'] == eligible(record)[0]
        if record['accepted']:
            accepted.append(target_sample(sample, record))
        else:
            rejected.append(record)
    assert len(accepted) >= math.ceil(.75 * 880), 'Too many rejected targets; inspect before training'
    destination = run / 'corrected_train.jsonl'
    if destination.exists():
        assert read_rows(destination) == accepted
    else:
        pilot.write_jsonl(destination, accepted)
        pilot.write_jsonl(run / 'quarantined_corrections.jsonl', rejected)
    audit = {'status': 'frozen', 'teacher_model': plan['teacher_model'], 'total': 880,
             'accepted': len(accepted), 'rejected': len(rejected),
             'labels': dict(Counter(shared.target_of(r['row'])['answer'] for r in accepted)),
             'decisions': dict(Counter(r['decision'] for r in records)),
             'source_dataset_unchanged': True, 'corrected_train_sha256': shared.digest(destination),
             'plan_sha256': shared.digest(run / 'plan.json'),
             'optimizer_steps': math.ceil(len(accepted)/8)}
    shared.write_json(run / 'correction_manifest.json', audit)
    status(run, 'freeze', status='complete', accepted=len(accepted), rejected=len(rejected))
    print(json.dumps(audit), flush=True)


def report(run):
    plan, _ = load_plan(run)
    old, before, after = (read_rows(BASELINE), read_rows(run/'test_before/predictions.jsonl'),
                          read_rows(run/'test/predictions.jsonl'))
    assert len(old) == len(before) == len(after) == 1082
    for a, b, c in zip(old, before, after):
        assert all(a[k] == b[k] == c[k] for k in ('sample_id', 'source_sample_id', 'expected'))
        assert all(b[k] == c[k] for k in ('prompt_sha256', 'input_tokens'))
    def comparison(a, b):
        first, last = summarize(a), summarize(b)
        return {'accuracy_delta_percentage_points': 100*(last['accuracy']-first['accuracy']),
                'corrected_errors': sum(not x['correct'] and y['correct'] for x,y in zip(a,b)),
                'new_errors': sum(x['correct'] and not y['correct'] for x,y in zip(a,b))}
    api_events = []
    for worker in range(WORKERS):
        api_events.extend(read_rows(run/'teachers'/str(worker)/'api_requests.jsonl'))
    tokens = Counter()
    for event in api_events:
        usage = event.get('usage') or {}
        tokens['input'] += usage.get('input_tokens', 0)
        tokens['output'] += usage.get('output_tokens', 0)
    model = {'adapter': str(run/'train/final'), 'base': str(BASE),
             'adapter_sha256': shared.digest(run/'train/final/adapter_model.safetensors'),
             'dtype': 'bfloat16', 'quantization': None, 'format': 'PEFT LoRA adapter; load with the specified base',
             'input_builder': str(ROOT/'scripts/qreflector_geometry_prompt.py'),
             'original_prompt': str(DATA/'prompt_template.txt'), 'image_max_side': 512,
             'source_adapter': str(ADAPTER), 'training_plan': str(run/'plan.json'),
             'test_metrics': str(run/'test/summary.json')}
    result = {'round1_original_input': summarize(old), 'round1_geometry_input': summarize(before),
              'round2_geometry_input': summarize(after), 'input_change_effect': comparison(old, before),
              'round2_training_effect': comparison(before, after),
              'same_test_inputs_labels_tokenization_before_after_training': True,
              'validation_used': False, 'selection': plan['evaluation']['checkpoint_policy'],
              'correction_manifest': shared.read_json(run/'correction_manifest.json'),
              'training': shared.read_json(run/'train_status.json'), 'api_usage': dict(tokens), 'model': model}
    shared.write_json(run/'model_manifest.json', model)
    shared.write_json(run/'comparison.json', result)
    lines = ['Q-Reflector: dynamic-coordinate input comparison and Sol round 2',
             'Model | Overall | Success recognition | Failure recognition | Correct / 1082']
    for name in ['round1_original_input', 'round1_geometry_input', 'round2_geometry_input']:
        v = result[name]
        lines.append(f"{name} | {v['accuracy']*100:.2f}% | {v['success_recognition_rate']*100:.2f}% | "
                     f"{v['failure_recognition_rate']*100:.2f}% | {v['correct']}/1082")
    lines += ['', 'Round-2 model: ' + model['adapter'], 'Required base: ' + model['base'],
              'Input builder: ' + model['input_builder'], 'Full results: ' + str(run/'comparison.json')]
    (run/'report.txt').write_text(chr(10).join(lines) + chr(10))
    status(run, 'report', status='complete')
    print(json.dumps(result, indent=2), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','teachers','freeze','report'])
    p.add_argument('--run-dir', type=Path, required=True)
    args=p.parse_args()
    if args.stage == 'prepare':
        prepare(args.run_dir.resolve())
        return
    with (args.run_dir / f'.{args.stage}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            globals()[args.stage](args.run_dir.resolve())
        except Exception as error:
            status(args.run_dir, args.stage, status='failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
