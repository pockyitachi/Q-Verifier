#!/usr/bin/env python3
"""One batched Sol correction round from v9 epoch 9; no validation selection.

All drafts come from the fixed initial student. Teacher correction completes
before one pass of assistant-only CE training. The final adapter alone is tested.
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
from qreflector_sft_bf16 import (
    BASE, PROCESSOR_BASE, check_bf16_base, environment, gpu_setup, read_rows, strict_structure,
)
from eval_gui_reflection import parse_verdict, summarize

ROOT, DATA, ADAPTER = shared.ROOT, pilot.DATA, pilot.ADAPTER
GPU = 'GPU-b754d897-e1c3-1431-5f50-36968f3cc1e5'
PILOT = ROOT / 'artifacts/evaluations/qreflector_v9_epoch9_teacher_correction100_20261008'
BASELINE = ROOT / 'artifacts/training/qreflector_bf16_v9_20261007_gpu7/test/predictions.jsonl'
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


def geometry(row, data=DATA, max_side=512):
    from PIL import Image
    images = []
    for c in row['messages'][0]['content']:
        if c['type'] != 'image':
            continue
        with Image.open(data / c['image']) as image:
            w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        images.append({'role': 'BEFORE' if not images else 'AFTER', 'original_width': w,
                       'original_height': h, 'sent_width': int(w*scale), 'sent_height': int(h*scale)})
    assert len(images) == 2
    before = images[0]
    args = context_of(row)['action'].get('arguments') or {}
    points = []
    for xk, yk in [('x', 'y'), ('x1', 'y1'), ('x2', 'y2')]:
        if xk not in args or yk not in args:
            continue
        x, y = args[xk], args[yk]
        assert isinstance(x, (int, float)) and isinstance(y, (int, float))
        nx, ny = x / before['original_width'], y / before['original_height']
        points.append({'keys': [xk, yk], 'original_pixels': [x, y],
                       'normalized_xy': [nx, ny],
                       'sent_image_pixels': [nx*before['sent_width'], ny*before['sent_height']],
                       'inside_original_image': 0 <= nx < 1 and 0 <= ny < 1})
    return {'coordinate_convention': 'Original BEFORE pixels; origin top-left, x right, y down',
            'images': images, 'action_points': points}


def teacher_request(row, draft, config):
    payload = RAW_TEACHER_REQUEST(row, draft, config)
    payload['instructions'] = TEACHER_INSTRUCTIONS
    payload['input'][0]['content'].insert(1, {'type': 'input_text', 'text':
        '=== IMAGE GEOMETRY (metadata, no outcome labels) ===\n' +
        json.dumps(geometry(row, Path(config['data']), config['image_max_side']))})
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
             'train.input_manifest.jsonl', 'prompt_template.txt', 'quarantine.jsonl']
    for name in names:
        assert shared.digest(DATA / name) == manifest['artifact_sha256'][name], name
    rows, labels = read_rows(DATA / 'train.jsonl'), read_rows(DATA / 'train.labels.jsonl')
    inputs = read_rows(DATA / 'train.inputs.jsonl')
    assert len(rows) == len(labels) == len(inputs) == 880
    assert shared.digest(ADAPTER / 'adapter_model.safetensors') == pilot.ADAPTER_SHA
    assert shared.read_json(PILOT / 'plan.json')['adapter'] == str(ADAPTER)
    assert shared.read_json(PILOT / 'plan.json')['pinned_sha256'][str(ADAPTER / 'adapter_model.safetensors')] == pilot.ADAPTER_SHA
    samples, image_hashes = [], {}
    for i, (row, label, inp) in enumerate(zip(rows, labels, inputs, strict=True)):
        assert row['messages'][0] == inp['messages'][0]
        assert label['answer'] == shared.target_of(row)['answer']
        assert label['sample_id'] not in manifest['quarantined_ids']
        for c in row['messages'][0]['content']:
            if c['type'] == 'image':
                image_hashes[c['image']] = shared.digest(DATA / c['image'])
        samples.append({'sample_id': label['sample_id'], 'train_line': i+1, 'row': row,
                        'student_messages_sha256': correction.digest_json(correction.student_messages(row)),
                        'expected': label['answer'], 'geometry': geometry(row)})
    by_id = {s['sample_id']: s for s in samples}
    imported = read_rows(PILOT / 'student_rollouts.jsonl')
    assert len(imported) == len({r['sample_id'] for r in imported}) == 100
    for draft in imported:
        sample = by_id[draft['sample_id']]
        assert draft['student_messages_sha256'] == sample['student_messages_sha256']
        assert draft['expected'] == sample['expected'] and draft['train_line'] == sample['train_line']
        draft['provenance'] = {'imported_same_initial_student': True, 'source': str(PILOT / 'student_rollouts.jsonl')}
    run.mkdir(parents=True)
    pilot.write_jsonl(run / 'samples.jsonl', samples)
    pilot.write_jsonl(run / 'student_rollouts.jsonl', imported)
    pinned = [Path(__file__).resolve(), ROOT / 'scripts/qreflector_sol_round1_gpu.py',
              ROOT / 'scripts/run_qreflector_sol_round1.py', ROOT / 'scripts/qreflector_luna_correction.py',
              ROOT / 'scripts/qreflector_opsd.py', ROOT / 'scripts/qreflector_teacher_pilot_v9.py',
              ROOT / 'scripts/qreflector_sft_bf16.py', ROOT / 'scripts/eval_gui_reflection.py',
              DATA / 'DATASET_MANIFEST.json', ADAPTER / 'adapter_model.safetensors',
              ADAPTER / 'adapter_config.json', BASE / 'config.json', BASE / 'model.safetensors.index.json',
              PILOT / 'plan.json', PILOT / 'student_rollouts.jsonl', BASELINE]
    pinned += [DATA / name for name in names]
    pinned += [ADAPTER / name for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                                         'processor_config.json']]
    plan = {'created_unix': time.time(), 'method': 'batched_student_response_correction_distillation',
            'not_token_kl_opd': True, 'round': 1, 'data': str(DATA), 'adapter': str(ADAPTER),
            'base': str(BASE), 'initialization': 'v9 BF16 LoRA, LR 2e-4, epoch 9 / step 990',
            'source_adapter_sha256': pilot.ADAPTER_SHA,
            'training_inputs': 880, 'imported_matching_student_drafts': 100, 'new_student_drafts': 780,
            'sampling': {'temperature': 1., 'top_p': 1., 'top_k': 0, 'max_new_tokens': 1024, 'seed': SEED},
            'image_max_side': 512, 'student_prompt_unchanged': True,
            'teacher_model': 'gpt-6.1-sol', 'teacher_reasoning': 'medium',
            'teacher_max_output_tokens': 4096, 'teacher_workers': WORKERS, 'max_api_calls_per_worker': 275,
            'teacher_instructions': TEACHER_INSTRUCTIONS, 'teacher_original_image_geometry': True,
            'teacher_privilege': 'Train-only fallible four-field reference and initial student draft',
            'known_annotation_holds': HOLD_IDS, 'minimum_accepted_fraction': .75,
            'training': {'epochs': 1, 'learning_rate': 5e-6, 'warmup_ratio': .1, 'scheduler': 'cosine',
                         'batch_size': 1, 'gradient_accumulation': 8, 'optimizer': 'AdamW',
                         'weight_decay': 0., 'max_gradient_norm': 1., 'max_sequence_length': 8192,
                         'max_target_tokens': 1024, 'assistant_only_ce': True, 'auxiliary_sft': False,
                         'rank': 16, 'alpha': 16, 'trainable_parameters': 43646976,
                         'save_every_steps': 25, 'seed': 42},
            'evaluation': {'validation': False, 'split': 'test', 'samples': 1082,
                           'checkpoint_policy': 'Predetermined final one-epoch correction checkpoint only',
                           'prompt': 'Unchanged v9', 'greedy': True, 'max_new_tokens': 1024,
                           'score': 'Final Yes/No; invalid outputs incorrect; pre/post/reason unscored'},
            'gpu_uuid': GPU, 'gpu_memory_cap_gib': CAP_GIB, 'reserve_free_gib': RESERVE_GIB,
            'minimum_start_free_gib': CAP_GIB + RESERVE_GIB,
            'baseline_predictions': str(BASELINE), 'test_sha256': manifest['artifact_sha256']['test.jsonl'],
            'samples_sha256': shared.digest(run / 'samples.jsonl'),
            'image_sha256': image_hashes, 'pinned_sha256': {str(p): shared.digest(p) for p in pinned},
            'process_policy': 'Spawn own children only; no signals, resets, services, or existing process changes'}
    shared.write_json(run / 'plan.json', plan)
    status(run, 'generate', status='prepared', completed=100, total=880)
    status(run, 'pipeline', status='prepared')
    print(json.dumps({'prepared': str(run), 'train': 880, 'reused_drafts': 100, 'validation': False}))


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
    before, after = read_rows(BASELINE), read_rows(run / 'test/predictions.jsonl')
    assert len(before) == len(after) == 1082
    for a,b in zip(before,after):
        assert all(a[k] == b[k] for k in ('sample_id','source_sample_id','expected','prompt_sha256','input_tokens'))
    first, last = summarize(before), summarize(after)
    api_events = []
    for worker in range(WORKERS):
        api_events.extend(read_rows(run/'teachers'/str(worker)/'api_requests.jsonl'))
    tokens = Counter()
    for event in api_events:
        usage=event.get('usage') or {};details=usage.get('input_tokens_details') or {}
        tokens['input']+=usage.get('input_tokens',0)
        tokens['output']+=usage.get('output_tokens',0)
        tokens['cached_input']+=details.get('cached_tokens',0)
        tokens['cache_write']+=details.get('cache_write_tokens',0)
    cost=((tokens['input']-tokens['cached_input']-tokens['cache_write'])*2+
          tokens['cached_input']*.1+tokens['cache_write']*2.5+tokens['output']*10)/1e6
    report = {'baseline_epoch9': first, 'sol_round1_final': last,
              'accuracy_delta_percentage_points': 100*(last['accuracy']-first['accuracy']),
              'corrected_errors': sum(not a['correct'] and b['correct'] for a,b in zip(before,after)),
              'new_errors': sum(a['correct'] and not b['correct'] for a,b in zip(before,after)),
              'same_test_inputs_labels_tokenization': True, 'validation_used': False,
              'selection': plan['evaluation']['checkpoint_policy'],
              'correction_manifest': shared.read_json(run / 'correction_manifest.json'),
              'training': shared.read_json(run / 'train_status.json'),
              'api_usage': dict(tokens), 'estimated_standard_api_usd':cost,
              'cost_includes_reasoning_and_cache_writes':True,
              'pricing_source':'https://developers.openai.com/api/docs/models/gpt-6.1-sol'}
    shared.write_json(run / 'comparison.json', report)
    status(run, 'report', status='complete')
    print(json.dumps(report, indent=2), flush=True)


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
