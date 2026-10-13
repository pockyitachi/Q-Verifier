#!/usr/bin/env python3
"""BF16 SFT step 333 on GUI-Reflection purposes with four-field output.

Preserves the official paired purposes and labels. Does not add global goals,
actual actions, history, or targets to model input. Uses original source PNGs.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from qreflector_sft_bf16 import (BASE, PROCESSOR_BASE, ROOT, check_bf16_base,
    digest, environment, gpu_setup, read_rows, strict_structure)
from eval_gui_reflection import gpu_snapshot, parse_verdict, save_json, summarize

DATA = ROOT / 'artifacts/datasets/gui_reflection_task_suite'
ADAPTER = ROOT / 'artifacts/training/qreflector_bf16_v6_20261004_gpu4/training/final'
GPU = 'GPU-f25fb1a7-2bc8-d28f-8264-315da6a4f5b5'
PROMPT = '''You are a mobile GUI reflector performing post-action verification.
Judge whether the specified action purpose was fulfilled by the action executed
between the BEFORE and AFTER screenshots. The action itself is implicit.
The action purpose below is the exact target to verify. Evaluate that purpose,
not whether the transition was useful for some other purpose or global task.
An intermediate step toward the specified purpose is not sufficient unless the
purpose itself describes that intermediate step. A screen change alone is not
sufficient. Do not replace the supplied purpose with a different purpose that
better matches the observed outcome.

Formulate the precondition and expected postcondition using the specified action
purpose and BEFORE screenshot. The expected postcondition should describe what
fulfillment of this purpose would look like; do not rewrite it to match failure.
Compare the BEFORE and AFTER screenshots and explain the observed evidence.
Both screenshots are provided together in this single post-action call.

=== Context ===
{context}

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Required Output ===
Return only one JSON object with these string fields in this order:
"precondition": the prerequisites for the specified purpose;
"postcondition": the expected outcome if the specified purpose is fulfilled;
"reason": a brief explanation grounded in the observed transition;
"answer": exactly "Yes" if the specified purpose was fulfilled, otherwise "No".
'''


def build_samples(pairs):
    samples = []
    for pair_id, pair in enumerate(pairs, 1):
        assert len(pair['images']) == 2
        for kind, label in [('positive', 'yes'), ('negative', 'no')]:
            purpose = pair[f'{kind}_action_purpose']
            assert isinstance(purpose, str) and purpose.strip()
            prompt = PROMPT.format(context=json.dumps({'action_purpose': purpose}, ensure_ascii=False, indent=2))
            samples.append({'sample_id': len(samples) + 1, 'pair_id': pair_id,
                'kind': kind, 'expected': label, 'purpose': purpose,
                'images': pair['images'], 'prompt': prompt,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    return samples


def validate_predictions(results, samples):
    assert len(results) <= len(samples)
    for row, sample in zip(results, samples):
        for key in ('sample_id', 'pair_id', 'kind', 'expected', 'purpose', 'images', 'prompt_sha256'):
            assert row[key] == sample[key], key
        assert row['predicted'] == parse_verdict(row['full_response'])
        assert row['correct'] == (row['predicted'] == sample['expected'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.output.resolve()
    assert ROOT / 'artifacts/evaluations' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    annotation = DATA / 'action_verification.jsonl'
    dataset = json.loads((DATA / 'dataset_manifest.json').read_text())
    assert digest(annotation) == dataset['annotation_sha256']
    pairs = read_rows(annotation)
    assert len(pairs) == 603
    samples = build_samples(pairs)
    assert len(samples) == 1206
    images = {item['name']: item for item in dataset['images']}
    assert set(images) == {name for row in pairs for name in row['images']}
    for name, item in images.items():
        assert digest(DATA / 'images' / name) == item['sha256'], name
    state = json.loads((ADAPTER / 'trainer_state.json').read_text())
    assert state['global_step'] == 333 and state['epoch'] == 3.0
    manifest = {'benchmark': dataset['benchmark_repo'], 'benchmark_revision': dataset['benchmark_revision'],
        'annotation_sha256': digest(annotation), 'image_manifest_sha256': digest(DATA / 'dataset_manifest.json'),
        'pairs': 603, 'judgments': 1206, 'yes': 603, 'no': 603,
        'base': str(BASE), 'base_source': json.loads((BASE.parent / 'source.json').read_text()),
        'adapter': str(ADAPTER), 'adapter_sha256': digest(ADAPTER / 'adapter_model.safetensors'),
        'adapter_config_sha256': digest(ADAPTER / 'adapter_config.json'), 'checkpoint_step': 333,
        'precision': 'BF16 base with trained LoRA adapter', 'prompt_template': PROMPT,
        'label_semantics': 'Fulfillment of supplied purpose; unchanged official positive/negative labels',
        'model_input': 'Two original PNGs and one purpose. No global goal, action coordinates, history, or label.',
        'output': ['precondition', 'postcondition', 'reason', 'answer'],
        'score': 'Final Yes/No only; invalid verdict counts as incorrect',
        'preprocessing': 'Native Qwen3-VL saved processor on original PNGs; no manual 512px downsampling',
        'protocol': 'Adapted structured prompt; not an exact reproduction of official direct Yes/No evaluation',
        'generation': {'max_new_tokens': 1024, 'do_sample': False, 'num_beams': 1},
        'gpu_uuid': GPU, 'memory_cap_gib': 40, 'minimum_startup_free_gib': 48,
        'evaluator_sha256': digest(Path(__file__)),
        'helper_sha256': {str(p): digest(p) for p in [ROOT / 'scripts/qreflector_sft_bf16.py', ROOT / 'scripts/eval_gui_reflection.py']},
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'Pillow']}}
    mf = run / 'manifest.json'
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest, 'Manifest drift'
    else:
        save_json(mf, manifest)
    (run / 'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
    (run / 'prompt_example.txt').write_text(samples[0]['prompt'])
    save_json(run / 'data_validation.json', {'pairs': 603, 'judgments': 1206,
        'verified_original_images': len(images), 'positive_negative_same_images': True,
        'positive_negative_purpose_preserved': True})
    if args.prepare_only:
        print(json.dumps({'prepared': str(run), 'judgments': len(samples), 'images_verified': len(images)}))
        return
    predictions = run / 'predictions.jsonl'
    results = read_rows(predictions) if predictions.exists() else []
    if predictions.exists():
        assert args.resume, 'Existing predictions require --resume'
    validate_predictions(results, samples)
    if len(results) == len(samples):
        save_json(run / 'summary.json', summarize(results))
        save_json(run / 'status.json', {'status': 'complete', 'completed': len(results), 'total': len(samples)})
        return
    environment(GPU, run / 'hf_cache')
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    gpu_setup(torch, 40)
    save_json(run / 'gpu_before.json', gpu_snapshot(GPU))
    save_json(run / 'status.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(results), 'total': 1206})
    processor = AutoProcessor.from_pretrained(str(ADAPTER), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    model.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    save_json(run / 'precision_validation.json', check_bf16_base(model))
    model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False).eval()
    loaded = get_peft_model_state_dict(model)
    with safe_open(str(ADAPTER / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        assert set(loaded) == set(saved.keys())
        for name, weight in loaded.items():
            assert torch.equal(weight.detach().cpu(), saved.get_tensor(name).to(weight.dtype)), name
    save_json(run / 'adapter_validation.json', {'all_saved_tensors_match_loaded': True, 'tensor_count': len(loaded)})
    del loaded
    started = time.monotonic()
    with predictions.open('a') as f:
        for sample in samples[len(results):]:
            while torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                torch.cuda.empty_cache()
                time.sleep(10)
            content = [{'type': 'text', 'text': sample['prompt']}]
            for name in sample['images']:
                with Image.open(DATA / 'images' / name) as opened:
                    content.append({'type': 'image', 'image': opened.convert('RGB')})
            begun = time.monotonic()
            with torch.inference_mode():
                inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', truncation=False)
                inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
                ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=1,
                    temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
                new_ids = ids[0, inputs['input_ids'].shape[-1]:]
                response = processor.decode(new_ids, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            verdict = parse_verdict(response)
            row = {k: v for k, v in sample.items() if k != 'prompt'}
            row.update(predicted=verdict, correct=verdict == sample['expected'], full_response=response,
                input_tokens=int(inputs['input_ids'].shape[-1]), output_tokens=len(new_ids),
                at_generation_limit=len(new_ids) >= 1024, valid_four_field_json=strict_structure(response),
                latency_seconds=time.monotonic() - begun)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            results.append(row)
            save_json(run / 'summary.json', summarize(results))
            save_json(run / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'completed': len(results), 'total': 1206, 'elapsed_seconds': time.monotonic() - started,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
            if len(results) <= 4 or len(results) % 50 == 0:
                print(f'{len(results)}/1206; {row["latency_seconds"]:.2f}s; tokens={len(new_ids)}', flush=True)
    validate_predictions(results, samples)
    assert len(results) == 1206
    save_json(run / 'result_validation.json', {'passed': True, 'judgments': len(results),
        'labels_and_purposes_match_official': True, 'predictions_sha256': digest(predictions)})
    save_json(run / 'status.json', {'status': 'complete', 'completed': 1206, 'total': 1206,
        'elapsed_seconds': time.monotonic() - started})
    save_json(run / 'gpu_after.json', gpu_snapshot(GPU))


if __name__ == '__main__':
    main()
