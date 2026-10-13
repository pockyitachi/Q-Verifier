#!/usr/bin/env python3
"""Evaluate GUI-Reflection on the existing AndroidWorld transition test set.

Uses the official model wrapper and unchanged test-set user text. Images use the
same 512-pixel longest-side resize as the existing Q-Verifier/GPT evaluators,
followed by the official GUI-Reflection image preprocessing. This is a matched-
input comparison, not a reproduction of the authors' Action Verification suite.
Only the selected GPU is visible; no external processes or services are changed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPU0_UUID = 'GPU-991ac45f-e9e9-1c25-590c-fb49ca752965'


def parse_verdict(response):
    text = response.strip()
    # Never infer No just because Yes is absent, or search inside the rationale.
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get('answer'), str):
            answer = obj['answer'].strip().lower()
            return answer if answer in ('yes', 'no') else None
    except (ValueError, TypeError):
        pass
    match = re.match(r'^(?:answer\s*:\s*)?[\s*\"\']*(yes|no)\b', text, re.I)
    return match.group(1).lower() if match else None


def summarize(results):
    counts = Counter((r['expected'], r['predicted']) for r in results)
    total = len(results)
    correct = sum(r['correct'] for r in results)
    output = {'total': total, 'correct': correct,
              'accuracy': correct / total if total else None,
              'invalid_outputs': sum(r['predicted'] is None for r in results),
              'confusion': {label: {pred: counts[(label, pred)] for pred in ['yes', 'no']}
                            | {'invalid': counts[(label, None)]} for label in ['yes', 'no']}}
    for label, name in [('yes', 'success'), ('no', 'failure')]:
        n = sum(r['expected'] == label for r in results)
        output[f'{name}_total'] = n
        output[f'{name}_recognition_rate'] = counts[(label, label)] / n if n else None
    output['false_rejection_rate'] = (counts[('yes', 'no')] / output['success_total']
                                      if output['success_total'] else None)
    output['false_acceptance_rate'] = (counts[('no', 'yes')] / output['failure_total']
                                      if output['failure_total'] else None)
    return output


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def gpu_snapshot(uuid):
    def query(field):
        return subprocess.check_output(['nvidia-smi', '-i', uuid, field, '--format=csv'], text=True)
    return {'gpu': query('--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu'),
            'processes': query('--query-compute-apps=pid,process_name,used_gpu_memory')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / '../testing_data_androidworld/test.jsonl')
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain')
    parser.add_argument('--official-code', type=Path, default=ROOT / 'artifacts/vendor/GUI_Reflection')
    parser.add_argument('--gpu-uuid', default=GPU0_UUID)
    parser.add_argument('--limit', type=int, default=1082)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--image-max-side', type=int, default=512)
    parser.add_argument('--max-tiles', type=int, default=6)
    parser.add_argument('--memory-cap-gib', type=float, default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['HF_HOME'] = str(ROOT / 'artifacts/cache/gui-reflection-hf')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    os.nice(10)

    from PIL import Image
    import torch
    sys.path.insert(0, str(args.official_code / 'internvl_chat'))
    from eval.infer import Model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    before = gpu_snapshot(args.gpu_uuid)
    free, capacity = torch.cuda.mem_get_info(0)
    cap = args.memory_cap_gib * 2**30
    if free < cap + 8 * 2**30:
        raise RuntimeError(f'Insufficient shared-GPU headroom: {free / 2**30:.1f} GiB free')
    torch.cuda.set_per_process_memory_fraction(cap / capacity, 0)

    raw = args.data.read_bytes()
    samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(samples) != 1082:
        raise ValueError(f'Expected 1082 samples, found {len(samples)}')
    if not 1 <= args.limit <= len(samples):
        raise ValueError('Invalid sample limit')
    prepared = []
    for i, sample in enumerate(samples):
        user = sample['messages'][0]
        assert user['role'] == 'user'
        text = '\n'.join(x['text'] for x in user['content'] if x['type'] == 'text')
        images = [x['image'] for x in user['content'] if x['type'] == 'image']
        if len(images) != 2 or any(not (args.data.parent / p).is_file() for p in images):
            raise ValueError(f'Missing image pair in sample {i + 1}')
        label = json.loads(sample['messages'][1]['content'][0]['text'])['answer'].lower()
        assert label in ('yes', 'no')
        prepared.append((text, images, label))
    provenance = json.loads(args.model.with_name(args.model.name + '.download.json').read_text())
    manifest = {
        'model_repo': provenance['repo_id'], 'model_revision': provenance['revision'],
        'official_code_revision': subprocess.check_output(['git', '-C', str(args.official_code), 'rev-parse', 'HEAD'], text=True).strip(),
        'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data': str(args.data.resolve()), 'data_sha256': hashlib.sha256(raw).hexdigest(),
        'dataset_size': len(samples), 'requested_samples': args.limit,
        'dataset_labels': dict(Counter(x[2] for x in prepared)),
        'prompt': 'unchanged test.jsonl user text; labels and reference explanations withheld',
        'image_policy': {'longest_side': args.image_max_side, 'resize': 'PIL LANCZOS',
                         'then': 'official dynamic_preprocess_v3', 'max_tiles': args.max_tiles},
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': args.max_new_tokens},
        'dtype': 'bfloat16', 'batch_size': 1, 'gpu_uuid': args.gpu_uuid,
        'memory_cap_gib': args.memory_cap_gib,
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'transformers', 'tokenizers', 'huggingface-hub', 'accelerate', 'peft', 'timm', 'Pillow']},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'manifest.json'
    predictions_path = args.output_dir / 'predictions.jsonl'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Output exists; resuming requires identical configuration')
    else:
        save_json(manifest_path, manifest)
    save_json(args.output_dir / 'gpu_before.json', before)
    results = []
    if predictions_path.exists():
        if not args.resume:
            raise RuntimeError('Predictions already exist')
        results = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
        for i, result in enumerate(results):
            assert result['sample_id'] == i + 1 and result['expected'] == prepared[i][2]
        if len(results) > args.limit:
            raise RuntimeError('More saved rows than requested')
        if len(results) == args.limit:
            print('All requested predictions are already present.', flush=True)
            return

    save_json(args.output_dir / 'status.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(results), 'total': args.limit})
    print(f'Loading {args.model} on physical GPU 0 ({args.gpu_uuid}); allocator cap {args.memory_cap_gib} GiB', flush=True)
    model = Model(str(args.model), max_num=args.max_tiles, auto=False)
    print(f'Loaded. Allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB.', flush=True)
    token_counts = {}
    original_generate = model.model.generate

    def measured_generate(*pos, **kwargs):
        token_counts['input'] = kwargs['input_ids'].shape[-1]
        generated = original_generate(*pos, **kwargs)
        token_counts['output'] = generated.shape[-1]
        return generated

    model.model.generate = measured_generate
    started = time.monotonic()
    with predictions_path.open('a') as output:
        for idx in range(len(results), args.limit):
            # Only release this process's unused cache if the shared card becomes crowded.
            if torch.cuda.mem_get_info()[0] < 8 * 2**30:
                torch.cuda.empty_cache()
                while torch.cuda.mem_get_info()[0] < 8 * 2**30:
                    print('Waiting for 8 GiB free GPU headroom; other processes left untouched.', flush=True)
                    time.sleep(10)
            text, paths, expected = prepared[idx]
            images = []
            for path in paths:
                with Image.open(args.data.parent / path) as img:
                    image = img.convert('RGB')
                if args.image_max_side and max(image.size) > args.image_max_side:
                    ratio = args.image_max_side / max(image.size)
                    image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)
                images.append(image)
            t0 = time.monotonic()
            with torch.inference_mode():
                response = model(text, images, {'max_new_tokens': args.max_new_tokens,
                                               'do_sample': False, 'num_beams': 1})
            torch.cuda.synchronize()
            predicted = parse_verdict(response)
            row = {'sample_id': idx + 1, 'expected': expected, 'predicted': predicted,
                   'correct': predicted == expected, 'full_response': response,
                   'images': paths, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'input_tokens': token_counts['input'], 'output_tokens': token_counts['output'],
                   'latency_seconds': time.monotonic() - t0}
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
            results.append(row)
            status = {'status': 'running', 'pid': os.getpid(), 'completed': len(results),
                      'total': args.limit, 'elapsed_seconds': time.monotonic() - started,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                      'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                      'updated_at_utc': datetime.now(timezone.utc).isoformat()}
            save_json(args.output_dir / 'status.json', status)
            save_json(args.output_dir / 'summary.json', summarize(results))
            if idx < 3 or (idx + 1) % 25 == 0:
                print(f"{idx + 1}/{args.limit}: {row['latency_seconds']:.2f}s; "
                      f"output={row['output_tokens']} tokens; parsed={predicted is not None}; "
                      f"peak={status['peak_allocated_gib']:.2f} GiB", flush=True)
    status['status'] = 'complete'
    save_json(args.output_dir / 'status.json', status)
    save_json(args.output_dir / 'gpu_after.json', gpu_snapshot(args.gpu_uuid))
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == '__main__':
    main()
