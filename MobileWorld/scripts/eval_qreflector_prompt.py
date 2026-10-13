#!/usr/bin/env python3
"""Evaluate a separately versioned prompt without changing frozen data or evaluators."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import string
import time
from pathlib import Path

from eval_gui_reflection import gpu_snapshot, parse_verdict, save_json, summarize
from qreflector_sft_bf16 import (
    BASE,
    DATA,
    PROCESSOR_BASE,
    ROOT,
    check_bf16_base,
    digest,
    environment,
    gpu_setup,
    load_images,
    read_rows,
    strict_structure,
    validate_dataset,
)
from run_qreflector_bf16 import verify_predictions


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render_user(row, original, replacement):
    """Preserve raw context text and ordered image paths; never read assistant labels."""
    fields = [field for _, field, _, _ in string.Formatter().parse(replacement) if field is not None]
    if fields != ['context'] or original.count('{context}') != 1:
        raise ValueError('Both templates must contain exactly one context placeholder')
    user = copy.deepcopy(row['messages'][0])
    if user['role'] != 'user' or [c['type'] for c in user['content']] != ['text', 'image', 'image']:
        raise ValueError('Expected one text and two ordered screenshots')
    prefix, suffix = original.split('{context}')
    old_text = user['content'][0]['text']
    if not old_text.startswith(prefix) or not suffix or not old_text.endswith(suffix):
        raise ValueError('Input does not match the frozen prompt template')
    context = old_text[len(prefix):-len(suffix)]
    parsed = json.loads(context)
    if set(parsed) != {'goal', 'action', 'history'}:
        raise ValueError('Unexpected context fields')
    user['content'][0]['text'] = replacement.format(context=context)
    return user, {
        'original_prompt_sha256': text_hash(old_text),
        'prompt_sha256': text_hash(user['content'][0]['text']),
        'context_sha256': text_hash(context),
        'image_paths': [c['image'] for c in user['content'] if c['type'] == 'image'],
    }


def prepare(args):
    validate_dataset(DATA, images=True)
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', args.gpu_uuid):
        raise ValueError('Supply exactly one full GPU UUID')
    run = args.output.resolve()
    if ROOT / 'artifacts/evaluations' not in run.parents:
        raise ValueError('Use a new artifacts/evaluations subdirectory')
    rows = read_rows(DATA / 'test.jsonl')
    baseline = verify_predictions(args.baseline.resolve(), rows)
    original = (DATA / 'prompt_template.txt').read_text()
    replacement = args.prompt.read_text()
    inputs = []
    for i, (row, old) in enumerate(zip(rows, baseline, strict=True), 1):
        user, record = render_user(row, original, replacement)
        assert record['original_prompt_sha256'] == old['prompt_sha256']
        assert user['content'][1:] == row['messages'][0]['content'][1:]
        assert record['prompt_sha256'] != record['original_prompt_sha256']
        inputs.append({'sample_id': i, **record})
    adapter = args.adapter.resolve()
    old_manifest = json.loads(args.baseline.with_name('manifest.json').read_text())
    assert old_manifest['adapter_sha256'] == digest(adapter / 'adapter_model.safetensors')
    assert old_manifest['data_sha256'] == digest(DATA / 'test.jsonl')
    assert old_manifest['image_max_side'] == 512 and old_manifest['max_new_tokens'] == 1024
    assert old_manifest['do_sample'] is False and old_manifest['dtype'] == 'bfloat16'
    files = [Path(__file__).resolve(), ROOT / 'scripts/qreflector_sft_bf16.py',
             ROOT / 'scripts/eval_gui_reflection.py', ROOT / 'scripts/run_qreflector_bf16.py',
             args.prompt.resolve(), DATA / 'FROZEN.json', DATA / 'test.jsonl',
             DATA / 'prompt_template.txt', args.baseline.resolve(), args.baseline.with_name('manifest.json').resolve(),
             BASE / 'config.json', BASE / 'model.safetensors.index.json',
             PROCESSOR_BASE / 'generation_config.json']
    files += [adapter / name for name in ('adapter_model.safetensors', 'adapter_config.json',
              'tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja')]
    plan = {
        'adapter': str(adapter), 'prompt_source': str(args.prompt.resolve()),
        'baseline_predictions': str(args.baseline.resolve()), 'baseline_summary': summarize(baseline),
        'gpu_uuid': args.gpu_uuid, 'memory_cap_gib': 40, 'reserve_free_gib': 8,
        'image_max_side': 512, 'max_new_tokens': 1024, 'max_sequence_length': 8192,
        'rows': 1082, 'dtype': 'bfloat16', 'do_sample': False, 'seed': 42,
        'pinned_files': {str(path): digest(path) for path in files},
        'prompt_sha256': text_hash(replacement), 'data_sha256': digest(DATA / 'test.jsonl'),
        'score': 'Final Yes/No only; invalid verdicts count as wrong.',
        'comparison_note': 'New condition-check instructions AND explicit execution-plus-progress criterion; not a pure order ablation.',
        'labels_changed': False, 'raw_context_and_ordered_images_unchanged': True,
        'process_policy': 'Only this new inference process is created. No signals to existing processes or GPU-setting changes.',
    }
    run.mkdir(parents=True, exist_ok=False)
    (run / 'prompt_template.txt').write_text(replacement)
    with (run / 'inputs_manifest.jsonl').open('x') as handle:
        for record in inputs:
            handle.write(json.dumps(record) + '\n')
    save_json(run / 'plan.json', plan)
    save_json(run / 'prompt_example.json', render_user(rows[0], original, replacement)[0])
    save_json(run / 'status.json', {'status': 'prepared', 'completed': 0, 'total': len(rows)})
    print(json.dumps({'prepared': str(run), 'rows': len(rows), 'baseline_correct': sum(r['correct'] for r in baseline)}))


def verify_plan(run):
    plan = json.loads((run / 'plan.json').read_text())
    for name, expected in plan['pinned_files'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Pinned input or script changed: {name}')
    if digest(run / 'prompt_template.txt') != plan['prompt_sha256']:
        raise ValueError('Run prompt snapshot changed')
    validate_dataset(DATA)
    return plan


def run_evaluation(args):
    run = args.output.resolve()
    plan = verify_plan(run)
    with (run / '.evaluation.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            evaluate_locked(run, plan, args.resume_results)
        except BaseException as error:
            status = json.loads((run / 'status.json').read_text())
            save_json(run / 'status.json', {**status, 'status': 'failed', 'pid': os.getpid(),
                                          'error': f'{type(error).__name__}: {error}'})
            raise


def evaluate_locked(run, plan, resume):
    predictions = run / 'predictions.jsonl'
    results = read_rows(predictions) if predictions.exists() else []
    if (predictions.exists() or (run / 'runtime.json').exists()) and not resume:
        raise ValueError('Run already started; use explicit resume or a new output directory')
    rows = read_rows(DATA / 'test.jsonl')
    inputs_manifest = read_rows(run / 'inputs_manifest.jsonl')
    original = (DATA / 'prompt_template.txt').read_text()
    replacement = (run / 'prompt_template.txt').read_text()
    users, records = [], []
    for i, row in enumerate(rows, 1):
        user, record = render_user(row, original, replacement)
        assert {'sample_id': i, **record} == inputs_manifest[i - 1]
        users.append(user)
        records.append(record)
    for i, result in enumerate(results):
        assert result['sample_id'] == i + 1 and result['prompt_sha256'] == records[i]['prompt_sha256']
        assert result['expected'] == json.loads(rows[i]['messages'][1]['content'][0]['text'])['answer'].lower()
        assert result['correct'] == (result['predicted'] == result['expected'])
    assert len(results) <= len(rows) == 1082
    environment(plan['gpu_uuid'], run / 'hf_cache')
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

    gpu_setup(torch, plan['memory_cap_gib'])
    save_json(run / 'gpu_before.json', gpu_snapshot(plan['gpu_uuid']))
    runtime = {'gpu_uuid': plan['gpu_uuid'], 'pid': os.getpid(),
               'plan_sha256': digest(run / 'plan.json'),
               'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')}}
    previous_runtime = run / 'runtime.json'
    if previous_runtime.exists():
        previous = json.loads(previous_runtime.read_text())
        assert previous['versions'] == runtime['versions'] and previous['plan_sha256'] == runtime['plan_sha256']
    save_json(previous_runtime, runtime)
    save_json(run / 'status.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(results), 'total': len(rows)})
    processor = AutoProcessor.from_pretrained(plan['adapter'], local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    base.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    save_json(run / 'precision_validation.json', check_bf16_base(base))
    model = PeftModel.from_pretrained(base, plan['adapter'], is_trainable=False)
    loaded = get_peft_model_state_dict(model)
    with safe_open(str(Path(plan['adapter']) / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        assert set(loaded) == set(saved.keys())
        for name, weight in loaded.items():
            assert torch.equal(weight.detach().cpu(), saved.get_tensor(name).to(weight.dtype)), name
    save_json(run / 'adapter_validation.json', {'all_saved_tensors_match_loaded': True, 'tensor_count': len(loaded)})
    del loaded
    model.eval()
    started = time.monotonic()
    with predictions.open('a') as output:
        for i in range(len(results), len(rows)):
            while torch.cuda.mem_get_info()[0] < plan['reserve_free_gib'] * 1024**3:
                torch.cuda.empty_cache()
                print('Waiting for GPU headroom; existing processes untouched', flush=True)
                time.sleep(10)
            user = copy.deepcopy(users[i])
            images = iter(load_images(rows[i], DATA, plan['image_max_side']))
            for item in user['content']:
                if item['type'] == 'image':
                    item['image'] = next(images)
            begun = time.monotonic()
            with torch.inference_mode():
                inputs = processor.apply_chat_template([user], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors='pt', truncation=False)
                assert inputs['input_ids'].shape[-1] + plan['max_new_tokens'] <= plan['max_sequence_length']
                inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
                output_ids = model.generate(**inputs, max_new_tokens=plan['max_new_tokens'],
                    do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                    repetition_penalty=1.0)
                new_ids = output_ids[0, inputs['input_ids'].shape[-1]:]
                response = processor.decode(new_ids, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            expected = json.loads(rows[i]['messages'][1]['content'][0]['text'])['answer'].lower()
            predicted = parse_verdict(response)
            record = {'sample_id': i + 1, 'expected': expected, 'predicted': predicted,
                'correct': predicted == expected, 'full_response': response,
                'valid_four_field_json': strict_structure(response), **records[i],
                'input_tokens': int(inputs['input_ids'].shape[-1]), 'output_tokens': len(new_ids),
                'at_generation_limit': len(new_ids) >= plan['max_new_tokens'],
                'latency_seconds': time.monotonic() - begun}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
            results.append(record)
            summary = summarize(results)
            summary['valid_four_field_json_rate'] = sum(r['valid_four_field_json'] for r in results) / len(results)
            save_json(run / 'summary.json', summary)
            save_json(run / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'completed': len(results), 'total': len(rows), 'elapsed_seconds': time.monotonic() - started,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
            if len(results) <= 3 or len(results) % 25 == 0:
                print(f'{len(results)}/1082 accuracy={summary["accuracy"]:.4f}; output={len(new_ids)} tokens; {record["latency_seconds"]:.2f}s', flush=True)
    baseline = verify_predictions(Path(plan['baseline_predictions']), rows)
    before, after = summarize(baseline), summarize(results)
    assert all(a['expected'] == b['expected'] and a['prompt_sha256'] == b['original_prompt_sha256']
               for a, b in zip(baseline, results, strict=True))
    save_json(run / 'comparison.json', {
        'epoch9_original_prompt': before, 'epoch9_condition_check_v1': after,
        'accuracy_delta_percentage_points': 100 * (after['accuracy'] - before['accuracy']),
        'corrected_errors': sum(not a['correct'] and b['correct'] for a, b in zip(baseline, results, strict=True)),
        'new_errors': sum(a['correct'] and not b['correct'] for a, b in zip(baseline, results, strict=True)),
        'score': plan['score'], 'comparison_note': plan['comparison_note'],
        'same_checkpoint_examples_labels_and_raw_context': True,
    })
    verify_plan(run)
    save_json(run / 'gpu_after.json', gpu_snapshot(plan['gpu_uuid']))
    save_json(run / 'status.json', {'status': 'complete', 'pid': os.getpid(),
        'completed': len(results), 'total': len(rows), 'elapsed_seconds': time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--adapter', type=Path, required=True)
    prep.add_argument('--prompt', type=Path, required=True)
    prep.add_argument('--baseline', type=Path, required=True)
    prep.add_argument('--gpu-uuid', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--resume-results', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run_evaluation}[args.mode](args)


if __name__ == '__main__':
    main()
