#!/usr/bin/env python3
"""Official AV prompt on our frozen 1082 AndroidWorld transitions, on GPU 7.

The action_purpose slot receives the existing action.description verbatim.
No purpose is generated from AFTER, reference conditions, reasons, or labels.
Original task-progress labels are retained: this is a transferred native-prompt
baseline, not the authors' native benchmark or a relabeling of our dataset.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from eval_gui_reflection import ROOT, gpu_snapshot, parse_verdict, save_json, summarize
from eval_gui_reflection_native import official_template, sha256_file

GPU_UUID = 'GPU-b754d897-e1c3-1431-5f50-36968f3cc1e5'
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
MODEL = ROOT / 'artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain'
OFFICIAL = ROOT / 'artifacts/vendor/GUI_Reflection'
PROMPT_SOURCE = OFFICIAL / 'internvl_chat/eval/eval_tasks/reflection_action_verification_infer.py'
GENERATION = {'max_new_tokens': 1024, 'do_sample': False, 'num_beams': 1,
              'temperature': 0.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def native_input(record, template):
    """Only the label-free user record enters prompt construction."""
    assert [m['role'] for m in record['messages']] == ['user']
    content = record['messages'][0]['content']
    text = '\n'.join(c['text'] for c in content if c['type'] == 'text')
    context = json.loads(text.split('=== Context ===', 1)[1].split('=== Image Reference ===', 1)[0])
    purpose = context['action']['description']
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError('Missing action description: no heuristic or label-based fallback')
    if '<image>' in purpose or '## Output:' in purpose:
        raise ValueError('Action description contains a prompt control marker')
    images = [c['image'] for c in content if c['type'] == 'image']
    assert len(images) == 2
    prompt = template.format(action_purpose=purpose)
    return {'purpose': purpose, 'images': images, 'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}


def prepare(output, resume):
    frozen = json.loads((DATA / 'DATASET_MANIFEST.json').read_text())
    for name in ['test.inputs.jsonl', 'test.labels.jsonl', 'image_sha256.json']:
        assert sha256_file(DATA / name) == frozen['artifact_sha256'][name], name
    template = official_template(PROMPT_SOURCE)
    records = read_rows(DATA / 'test.inputs.jsonl')
    labels = read_rows(DATA / 'test.labels.jsonl')
    assert len(records) == len(labels) == 1082
    assert Counter(row['answer'] for row in labels) == {'Yes': 613, 'No': 469}
    prepared = [dict(sample_id=i, **native_input(row, template)) for i, row in enumerate(records, 1)]
    # Check that this comparison retains the v6 baseline's descriptions, images,
    # order and labels. Only the user prompt/interface changes.
    previous = ROOT / '../datasets/condition_generation_v6_user_reviewed'
    previous_inputs = read_rows(previous / 'test.inputs.jsonl')
    assert len(previous_inputs) == len(prepared)
    for current, old in zip(prepared, previous_inputs, strict=True):
        old_input = native_input(old, template)
        assert current['purpose'] == old_input['purpose']
        assert [(DATA / p).resolve() for p in current['images']] == [
            (previous / p).resolve() for p in old_input['images']]
    previous_labels = [json.loads(row['messages'][1]['content'][0]['text'])['answer']
                       for row in read_rows(previous / 'test.jsonl')]
    assert previous_labels == [row['answer'] for row in labels]
    image_manifest = json.loads((DATA / 'image_sha256.json').read_text())
    for name in {name for case in prepared for name in case['images']}:
        assert sha256_file(DATA / name) == image_manifest[name]['sha256'], name

    provenance = json.loads(MODEL.with_name(MODEL.name + '.download.json').read_text())
    manifest = {
        'model_repo': provenance['repo_id'], 'model_revision': provenance['revision'],
        'model_path': str(MODEL), 'data': str(DATA), 'dataset_size': 1082,
        'dataset_labels': dict(Counter(row['answer'] for row in labels)),
        'input_sha256': sha256_file(DATA / 'test.inputs.jsonl'),
        'labels_sha256': sha256_file(DATA / 'test.labels.jsonl'),
        'dataset_manifest_sha256': sha256_file(DATA / 'DATASET_MANIFEST.json'),
        'evaluator_sha256': sha256_file(Path(__file__)),
        'helpers_sha256': {name: sha256_file(ROOT / 'scripts' / name)
                          for name in ['eval_gui_reflection.py', 'eval_gui_reflection_native.py']},
        'official_code_revision': subprocess.check_output(
            ['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip(),
        'prompt_source_sha256': sha256_file(PROMPT_SOURCE), 'prompt_template': template,
        'purpose_mapping': 'context.action.description verbatim; no LLM rewrite or AFTER access',
        'purpose_source': 'Existing action description proxy, not an official benchmark purpose annotation',
        'model_input': 'Two ordered screenshots and action_purpose only; goal/history/conditions/reason/label omitted',
        'label_policy': 'Original AndroidWorld task-aware Yes/No labels unchanged; no purpose-only relabeling',
        'v6_descriptions_images_order_labels_match': True,
        'image_policy': {'longest_side': 512, 'resize': 'PIL LANCZOS',
                         'then': 'official dynamic_preprocess_v3', 'max_tiles': 6},
        'generation': GENERATION, 'dtype': 'bfloat16', 'batch_size': 1,
        'gpu_index': 7, 'gpu_uuid': GPU_UUID, 'torch_memory_cap_gib': 32,
        'minimum_start_free_gib': 56, 'minimum_runtime_free_gib': 24,
        'versions': {p: importlib.metadata.version(p) for p in
                     ['torch', 'torchvision', 'transformers', 'tokenizers', 'peft', 'Pillow']},
    }
    target = output / 'manifest.json'
    if target.exists():
        assert resume and json.loads(target.read_text()) == manifest, 'Resume manifest mismatch'
    else:
        assert not (output / 'predictions.jsonl').exists(), 'Refusing to overwrite existing predictions'
        save_json(target, manifest)
    text = ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in prepared)
    target = output / 'native_inputs.jsonl'
    if target.exists():
        assert target.read_text() == text, 'Prepared native inputs changed'
    else:
        target.write_text(text)
    (output / 'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'native_prompt_template.txt').write_text(template)
    save_json(output / 'input_audit.json', {'passed': True, 'samples': 1082,
        'purpose_source': 'action.description verbatim', 'missing_descriptions': 0,
        'goal_history_labels_reference_answers_not_in_prompt': True,
        'v6_descriptions_images_order_labels_match': True,
        'native_inputs_sha256': sha256_file(output / 'native_inputs.jsonl')})
    return prepared, labels


def verify_saved(rows, prepared, labels):
    assert len(rows) <= len(prepared)
    for i, row in enumerate(rows):
        assert row['sample_id'] == i + 1
        assert row['source_sample_id'] == labels[i]['sample_id']
        assert row['expected'] == labels[i]['answer'].lower()
        assert row['prompt_sha256'] == prepared[i]['prompt_sha256']
        assert row['predicted'] == parse_verdict(row['full_response'])
        assert row['correct'] == (row['predicted'] == row['expected'])


def metrics(rows):
    summary = summarize(rows)
    summary['official_prefix_accuracy'] = sum(row['official_correct'] for row in rows) / len(rows)
    summary['response_counts'] = dict(Counter(row['predicted'] or 'invalid' for row in rows))
    return summary


def evaluate(output, prepared, labels):
    predictions = output / 'predictions.jsonl'
    results = read_rows(predictions) if predictions.exists() else []
    verify_saved(results, prepared, labels)
    if len(results) == len(prepared):
        save_json(output / 'summary.json', metrics(results))
        save_json(output / 'status.json', {'status': 'complete', 'completed': len(results), 'total': 1082})
        return
    while True:
        free = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
        if free >= 56:
            break
        save_json(output / 'status.json', {'status': 'waiting_for_gpu_headroom', 'pid': os.getpid(),
            'free_gib': free, 'required_free_gib': 56, 'completed': len(results), 'total': 1082})
        time.sleep(30)
    os.environ.update({'CUDA_VISIBLE_DEVICES': GPU_UUID, 'HF_HOME': str(output / 'hf_cache'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0', 'USE_FLAX': '0',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
    os.nice(10)
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    assert torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info(0)
    assert free >= 56 * 1024**3, 'GPU headroom fell before loading; exiting only this new process'
    torch.cuda.set_per_process_memory_fraction(32 * 1024**3 / total, 0)
    save_json(output / 'gpu_before.json', gpu_snapshot(GPU_UUID))
    save_json(output / 'status.json', {'status': 'loading', 'pid': os.getpid(),
              'completed': len(results), 'total': 1082})
    sys.path.insert(0, str(OFFICIAL / 'internvl_chat'))
    from eval.infer import Model
    print('Loading GUI-Reflection 8B on physical GPU 7; 32 GiB allocator cap', flush=True)
    model = Model(str(MODEL), max_num=6, auto=False)
    assert all(p.dtype == torch.bfloat16 for p in model.model.parameters() if p.is_floating_point())
    counts = {}
    generate = model.model.generate

    def counted_generate(*args, **kwargs):
        generated = generate(*args, **kwargs)
        counts.update(input_tokens=int(kwargs['input_ids'].shape[-1]),
                      output_tokens=int(generated.shape[-1]))
        return generated

    model.model.generate = counted_generate
    started = time.monotonic()
    with predictions.open('a') as output_file:
        for i in range(len(results), len(prepared)):
            if torch.cuda.mem_get_info(0)[0] < 24 * 1024**3:
                raise RuntimeError('GPU has under 24 GiB free; exiting only this evaluation process')
            sample = prepared[i]
            images = []
            for path in sample['images']:
                with Image.open(DATA / path) as original:
                    image = original.convert('RGB')
                if max(image.size) > 512:
                    ratio = 512 / max(image.size)
                    image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)
                images.append(image)
            begun = time.monotonic()
            with torch.inference_mode():
                response = model(sample['prompt'], images, dict(GENERATION))
            torch.cuda.synchronize()
            predicted = parse_verdict(response)
            # The reference label is only consulted after inference.
            expected = labels[i]['answer'].lower()
            row = {k: sample[k] for k in ['sample_id', 'purpose', 'images', 'prompt_sha256']}
            row.update(source_sample_id=labels[i]['sample_id'], expected=expected, predicted=predicted,
                correct=predicted == expected, full_response=response,
                official_correct=response.strip().lower().startswith(expected),
                latency_seconds=time.monotonic() - begun, **counts)
            output_file.write(json.dumps(row, ensure_ascii=False) + '\n')
            output_file.flush()
            os.fsync(output_file.fileno())
            results.append(row)
            save_json(output / 'summary.json', metrics(results))
            status = {'status': 'running', 'pid': os.getpid(), 'completed': len(results), 'total': 1082,
                'elapsed_seconds': time.monotonic() - started,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                'updated_at_utc': datetime.now(timezone.utc).isoformat()}
            save_json(output / 'status.json', status)
            if len(results) <= 3 or len(results) % 25 == 0:
                print(f'{len(results)}/1082; output={counts["output_tokens"]} tokens; '
                      f'latency={row["latency_seconds"]:.2f}s; accuracy={metrics(results)["accuracy"]:.4f}', flush=True)
    verify_saved(read_rows(predictions), prepared, labels)
    save_json(output / 'result_validation.json', {'passed': True, 'checked_samples': 1082,
        'predictions_sha256': sha256_file(predictions), 'labels_unchanged': True})
    save_json(output / 'status.json', {**status, 'status': 'complete'})
    save_json(output / 'gpu_after.json', gpu_snapshot(GPU_UUID))
    print(json.dumps(metrics(results), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert ROOT / 'artifacts/evaluations/gui_reflection' in output.parents
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepared, labels = prepare(output, args.resume)
            if args.prepare_only:
                save_json(output / 'status.json', {'status': 'prepared', 'total': 1082, 'completed': 0})
                print('Prepared 1082 native prompts; no model loaded.', flush=True)
            else:
                evaluate(output, prepared, labels)
        except Exception as error:
            prior = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
            save_json(output / 'status.json', {**prior, 'status': 'failed', 'pid': os.getpid(),
                'error': f'{type(error).__name__}: {error}', 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == '__main__':
    main()
