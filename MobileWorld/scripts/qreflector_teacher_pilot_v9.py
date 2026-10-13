#!/usr/bin/env python3
"""100 shared on-policy drafts, two API correction teachers; no weight updates.

Only the frozen v9 training split is sampled. References go to correction
teachers, never into student generation. This is not a held-out accuracy test.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import qreflector_luna_correction as correction
import qreflector_opsd as shared
from qreflector_sft_bf16 import (
    BASE, PROCESSOR_BASE, check_bf16_base, environment, gpu_setup,
    read_rows, strict_structure,
)
from eval_gui_reflection import parse_verdict

ROOT = shared.ROOT
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
ADAPTER = ROOT / 'artifacts/training/qreflector_bf16_v9_20261007_gpu7/train/checkpoint-990'
ADAPTER_SHA = '5a61da5b07a95d3faf8d20f05a25497ce7b70f6fe17eaa6cac1a8e55f3ee6cc9'
TEACHERS = ('gpt-6-luna', 'gpt-6.1-sol')
RATES = {'gpt-6-luna': (0.10, 0.01, 0.50), 'gpt-6.1-sol': (2.00, 0.10, 10.00)}
GPU = 'GPU-fb63c19e-9ce2-10d1-3bfe-302cd3125430'
SEED = 20261008
TEACHER_INSTRUCTIONS = correction.TEACHER_INSTRUCTIONS + '''
The original v9 observation defines the task and output semantics. In particular:
Preconditions are minimal action-specific prerequisites grounded in BEFORE,
not a paraphrase of action intent or a description of AFTER. Do not invent
invisible prerequisites. Expected postconditions describe only this action's
local effect, not later task steps, and must not be rewritten to match AFTER.
An incorrect action description is not ground truth about the actual operation.
AFTER is the primary outcome evidence; use BEFORE to establish the change.
Keep all three explicit reason labels: Precondition check:, Postcondition check:,
Task progress:. A useful preparatory step can succeed; an incorrect target or
value does not count as partial progress. Correct errors in reference conditions
and explanations too. Do not copy them merely because they are provided.
'''


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def select_indices(labels):
    """Predeclared proportional stratification, independent of model errors."""
    rng = random.Random(SEED)
    chosen = []
    for answer, count in [('Yes', 58), ('No', 42)]:
        pool = [i for i, row in enumerate(labels) if row['answer'] == answer]
        chosen.extend(rng.sample(pool, count))
    rng.shuffle(chosen)
    return chosen


def prepare(run):
    if run.exists():
        raise ValueError('Use a new output directory; never overwrite a pilot')
    manifest = shared.read_json(DATA / 'DATASET_MANIFEST.json')
    assert manifest['status'] == 'frozen'
    assert manifest['dataset_version'] == 'condition_generation_v9_after_evidence'
    names = ['train.jsonl', 'train.inputs.jsonl', 'train.labels.jsonl',
             'train.line_map.jsonl', 'quarantine.jsonl', 'prompt_template.txt']
    for name in names:
        assert shared.digest(DATA / name) == manifest['artifact_sha256'][name], name
    rows = read_rows(DATA / 'train.jsonl')
    inputs = read_rows(DATA / 'train.inputs.jsonl')
    labels = read_rows(DATA / 'train.labels.jsonl')
    line_map = read_rows(DATA / 'train.line_map.jsonl')
    assert len(rows) == len(inputs) == len(labels) == len(line_map) == 880
    assert Counter(r['answer'] for r in labels) == {'Yes': 512, 'No': 368}
    assert shared.digest(ADAPTER / 'adapter_model.safetensors') == ADAPTER_SHA
    samples, image_hashes = [], {}
    for i in select_indices(labels):
        assert rows[i]['messages'][0] == inputs[i]['messages'][0]
        assert shared.target_of(rows[i])['answer'] == labels[i]['answer']
        assert labels[i]['sample_id'] == line_map[i]['sample_id']
        assert labels[i]['sample_id'] not in manifest['quarantined_ids']
        paths = [c['image'] for c in rows[i]['messages'][0]['content'] if c['type'] == 'image']
        assert len(paths) == 2
        for path in paths:
            image_hashes[path] = shared.digest(DATA / path)
        samples.append({'sample_id': labels[i]['sample_id'], 'train_line': i + 1,
                        'reference': shared.target_of(rows[i]),
                        'student_messages_sha256': correction.digest_json(correction.student_messages(rows[i])),
                        'row': rows[i]})
    run.mkdir(parents=True)
    write_jsonl(run / 'samples.jsonl', samples)
    pinned = [Path(__file__), ROOT / 'scripts/qreflector_luna_correction.py',
              ROOT / 'scripts/qreflector_opsd.py', ROOT / 'scripts/qreflector_sft_bf16.py',
              ROOT / 'scripts/eval_gui_reflection.py', ADAPTER / 'adapter_model.safetensors',
              ADAPTER / 'adapter_config.json', DATA / 'DATASET_MANIFEST.json']
    pinned += [DATA / name for name in names]
    plan = {'data': str(DATA), 'adapter': str(ADAPTER), 'base': str(BASE),
            'base_description': 'v9 BF16 LoRA; LR 2e-4; epoch 9; step 990',
            'seed': SEED, 'samples': 100, 'labels': {'Yes': 58, 'No': 42},
            'sample_sha256': shared.digest(run / 'samples.jsonl'),
            'pinned_sha256': {str(p): shared.digest(p) for p in pinned},
            'image_sha256': image_hashes, 'gpu_uuid': GPU, 'memory_cap_gib': 32,
            'student': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
                        'max_new_tokens': 1024, 'image_max_side': 512},
            'teachers': list(TEACHERS), 'teacher_reasoning': 'medium',
            'teacher_max_output_tokens': 4096, 'max_api_calls_per_model': 120,
            'teacher_instructions': TEACHER_INSTRUCTIONS,
            'reference_policy': 'Train labels and references supplied to teachers only; fallible annotations',
            'weight_updates': False, 'test_or_validation_inputs_used': False,
            'prices_usd_per_million': {k: dict(zip(('input', 'cached_input', 'output'), v))
                                      for k, v in RATES.items()},
            'created_unix': time.time()}
    shared.write_json(run / 'plan.json', plan)
    shared.write_json(run / 'status_student.json', {'status': 'prepared', 'completed': 0})
    print(json.dumps({'prepared': str(run), 'samples': 100, 'labels': plan['labels']}), flush=True)


def load_plan(run):
    plan = shared.read_json(run / 'plan.json')
    assert shared.digest(run / 'samples.jsonl') == plan['sample_sha256']
    for path, expected in plan['pinned_sha256'].items():
        assert shared.digest(path) == expected, f'Pinned file changed: {path}'
    return plan, read_rows(run / 'samples.jsonl')


def student(run):
    plan, samples = load_plan(run)
    environment(plan['gpu_uuid'], run / 'hf_cache')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    # Abort before loading if someone has occupied the previously empty device.
    occupied = subprocess.check_output(['nvidia-smi', '-i', plan['gpu_uuid'],
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('Pilot GPU is no longer empty; no existing process is changed')
    gpu_setup(torch, plan['memory_cap_gib'])
    for path, expected in plan['image_sha256'].items():
        assert shared.digest(DATA / path) == expected
    destination = run / 'student_rollouts.jsonl'
    records = read_rows(destination) if destination.exists() else []
    assert len(records) <= len(samples)
    for sample, record in zip(samples, records):
        assert sample['sample_id'] == record['sample_id']
        assert sample['student_messages_sha256'] == record['student_messages_sha256']
    if len(records) == len(samples):
        return
    shared.write_json(run / 'status_student.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(records)})
    processor = AutoProcessor.from_pretrained(str(ADAPTER), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    model.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    precision = check_bf16_base(model)
    model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
    loaded = get_peft_model_state_dict(model)
    with safe_open(str(ADAPTER / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        assert set(loaded) == set(saved.keys())
        for name, weight in loaded.items():
            assert torch.equal(weight.detach().cpu(), saved.get_tensor(name).to(weight.dtype)), name
    del loaded
    shared.write_json(run / 'model_validation.json', {**precision, 'adapter_tensors_verified': True})
    model.eval()
    begun = time.monotonic()
    for index in range(len(records), len(samples)):
        sample = samples[index]
        if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
            raise RuntimeError('Shared GPU headroom fell below 24 GiB; exiting own pilot only')
        torch.manual_seed(SEED + index)
        user_only = correction.student_messages(sample['row'])
        assert len(user_only) == 1 and user_only[0]['role'] == 'user'
        assert correction.digest_json(user_only) == sample['student_messages_sha256']
        messages = shared.materialize_messages(user_only, DATA, 512)
        started = time.monotonic()
        with torch.inference_mode():
            inputs = shared.encode_messages(processor, messages)
            inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
            prompt_length = inputs['input_ids'].shape[-1]
            assert prompt_length + 1024 <= 8192
            generated = model.generate(**inputs, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, max_new_tokens=1024, num_beams=1, repetition_penalty=1.0)
            new_ids = generated[0, prompt_length:]
            text = processor.decode(new_ids, skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        record = {'sample_id': sample['sample_id'], 'train_line': sample['train_line'],
                  'student_messages_sha256': sample['student_messages_sha256'],
                  'full_response': text, 'expected': sample['reference']['answer'],
                  'predicted': parse_verdict(text), 'valid_four_field_json': strict_structure(text),
                  'input_tokens': int(prompt_length), 'output_tokens': len(new_ids),
                  'at_generation_limit': len(new_ids) >= 1024,
                  'latency_seconds': time.monotonic() - started, 'seed': SEED + index}
        correction.append_json(destination, record)
        shared.write_json(run / 'status_student.json', {'status': 'complete' if index == 99 else 'running',
            'pid': os.getpid(), 'completed': index + 1, 'total': 100,
            'elapsed_seconds': time.monotonic() - begun,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
        if index < 3 or (index + 1) % 10 == 0:
            print(f'Student {index + 1}/100; {record["latency_seconds"]:.2f}s', flush=True)
    assert shared.digest(ADAPTER / 'adapter_model.safetensors') == ADAPTER_SHA


def teacher_worker(run, plan, samples, drafts, model, limit):
    output = run / model
    output.mkdir(exist_ok=True)
    config = {'data': str(DATA), 'teacher_model': model, 'teacher_reasoning': 'medium',
              'teacher_max_output_tokens': 4096, 'max_api_calls': 120, 'image_max_side': 512}
    # Reuse secure retry/usage/cache code; only replace its request builder locally.
    client = correction.LunaClient(output, config, ROOT / '.env')
    records = []
    destination = output / 'corrections.jsonl'
    old = read_rows(destination) if destination.exists() else []
    for i, record in enumerate(old):
        assert record['sample_id'] == samples[i]['sample_id']
        assert record['draft_sha256'] == correction.digest_json(drafts[i]['full_response'])
    records.extend(old)
    for i in range(len(old), limit):
        sample, draft = samples[i], drafts[i]
        result, cached = client.correct(sample['row'], draft['full_response'])
        record = {'sample_id': sample['sample_id'],
                  'draft_sha256': correction.digest_json(draft['full_response']),
                  'student_messages_sha256': sample['student_messages_sha256'],
                  'expected': sample['reference']['answer'], 'student_predicted': draft['predicted'],
                  **result}
        correction.append_json(destination, record)
        records.append(record)
        shared.write_json(output / 'status.json', {'status': 'complete' if len(records) == 100 else 'running',
            'completed': len(records), 'total': 100, 'api_calls': client.calls,
            'input_tokens': client.input_tokens, 'output_tokens': client.output_tokens})
        if len(records) <= 2 or len(records) % 10 == 0:
            print(f'{model}: {len(records)}/100; {result["decision"]}; {result["elapsed_seconds"]:.1f}s', flush=True)
    return len(records)


def teachers(run, limit):
    plan, samples = load_plan(run)
    drafts = read_rows(run / 'student_rollouts.jsonl')
    assert len(drafts) >= limit
    for sample, draft in zip(samples, drafts):
        assert sample['sample_id'] == draft['sample_id']
        assert sample['student_messages_sha256'] == draft['student_messages_sha256']
    # Both threads use identical prompt content; only model differs.
    correction.TEACHER_INSTRUCTIONS = TEACHER_INSTRUCTIONS
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(teacher_worker, run, plan, samples, drafts, model, limit)
                   for model in TEACHERS]
        for future in futures:
            future.result()
    report(run)


def report(run):
    plan, samples = load_plan(run)
    drafts = read_rows(run / 'student_rollouts.jsonl')
    result = {'base': plan['base_description'], 'student_samples': len(drafts),
              'student_label_agreement': sum(r['predicted'] == r['expected'].lower() for r in drafts),
              'student_valid_json': sum(r['valid_four_field_json'] for r in drafts),
              'weight_updates': False, 'models': {},
              'caveat': 'Teacher sees fallible train references; label agreement is not held-out accuracy or a semantic quality assessment.'}
    for model in TEACHERS:
        output = run / model
        records = read_rows(output / 'corrections.jsonl') if (output / 'corrections.jsonl').exists() else []
        events = read_rows(output / 'api_requests.jsonl') if (output / 'api_requests.jsonl').exists() else []
        inp = sum((r.get('usage') or {}).get('input_tokens', 0) for r in events)
        out = sum((r.get('usage') or {}).get('output_tokens', 0) for r in events)
        cached = sum(((r.get('usage') or {}).get('input_tokens_details') or {}).get('cached_tokens', 0) for r in events)
        price_in, price_cache, price_out = RATES[model]
        counts = Counter()
        for r in records:
            parsed = (r.get('parsed') or {}).get('corrected_response') or {}
            verdict = parsed.get('answer', '')
            counts['label_agreement'] += verdict == r['expected']
            counts['accepted'] += r['accepted']
            counts['fixed_student_label'] += r['student_predicted'] != r['expected'].lower() and verdict == r['expected']
            counts['changed_correct_student_label'] += r['student_predicted'] == r['expected'].lower() and verdict != r['expected']
            counts['three_reason_checks'] += all(k in parsed.get('reason', '') for k in
                ('Precondition check:', 'Postcondition check:', 'Task progress:'))
        result['models'][model] = {'completed': len(records), **counts,
            'decisions': dict(Counter(r['decision'] for r in records)),
            'input_tokens': inp, 'cached_input_tokens': cached, 'output_tokens': out,
            'estimated_api_cost_usd': ((inp-cached)*price_in+cached*price_cache+out*price_out)/1e6,
            'api_calls': sum(e['event'] == 'request_started' for e in events)}
    shared.write_json(run / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'student', 'teachers', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    assert 1 <= args.limit <= 100
    if args.mode == 'prepare':
        prepare(args.output)
        return
    with (args.output / f'{args.mode}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'student':
            student(args.output)
        elif args.mode == 'teachers':
            teachers(args.output, args.limit)
        else:
            report(args.output)


if __name__ == '__main__':
    main()
