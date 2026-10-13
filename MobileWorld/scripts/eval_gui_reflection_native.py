#!/usr/bin/env python3
"""Frozen-model comparison on the official GUI-Reflection verification suite.

The two screenshots and action-purpose prompt are shared. GUI-Reflection uses
its unmodified official wrapper; Q-Reflector uses its configured 4-bit base and
LoRA with native Qwen3-VL preprocessing. No extra transition fields are invented.
"""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from eval_gui_reflection import GPU0_UUID, ROOT, gpu_snapshot, parse_verdict, save_json, summarize


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def official_template(source):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt_format' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Official verification prompt not found')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['gui-reflection', 'qreflector'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'artifacts/datasets/gui_reflection_task_suite')
    parser.add_argument('--adapter', type=Path, default=ROOT / '../GRPO/qwen3_vl_distilled_dpo_grpo_8b/final')
    parser.add_argument('--gpu-uuid', default=GPU0_UUID)
    parser.add_argument('--limit-pairs', type=int, default=603)
    parser.add_argument('--memory-cap-gib', type=float, default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update({'CUDA_VISIBLE_DEVICES': args.gpu_uuid,
                       'HF_HOME': str(ROOT / 'artifacts/cache/native-reflection-hf'),
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'USE_TF': '0', 'USE_FLAX': '0', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
    os.nice(10)
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    free, capacity = torch.cuda.mem_get_info()
    if free < (args.memory_cap_gib + 8) * 2**30:
        raise RuntimeError('Insufficient shared-GPU memory headroom')
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 2**30 / capacity)

    official = ROOT / 'artifacts/vendor/GUI_Reflection'
    source = official / 'internvl_chat/eval/eval_tasks/reflection_action_verification_infer.py'
    template = official_template(source)
    annotation = args.data_dir / 'action_verification.jsonl'
    rows = [json.loads(line) for line in annotation.read_text().splitlines() if line.strip()]
    dataset = json.loads((args.data_dir / 'dataset_manifest.json').read_text())
    if len(rows) != 603 or not 1 <= args.limit_pairs <= 603:
        raise ValueError('Unexpected evaluation size')
    assert sha256_file(annotation) == dataset['annotation_sha256']
    assert dataset['image_count'] == 1206
    prepared = []
    for i, row in enumerate(rows[:args.limit_pairs]):
        for kind, expected in [('positive', 'yes'), ('negative', 'no')]:
            purpose = row[f'{kind}_action_purpose']
            prompt = template.format(action_purpose=purpose)
            if len(row['images']) != 2:
                raise ValueError('Expected two screenshots')
            prepared.append({'sample_id': len(prepared) + 1, 'pair_id': i + 1,
                             'kind': kind, 'expected': expected, 'purpose': purpose,
                             'prompt': prompt, 'images': row['images']})

    generation = {'max_new_tokens': 1024, 'do_sample': False, 'num_beams': 1,
                  'temperature': 0.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0}
    manifest = {'model': args.model, 'benchmark': dataset['benchmark_repo'],
                'benchmark_revision': dataset['benchmark_revision'],
                'annotation_sha256': dataset['annotation_sha256'],
                'dataset_manifest_sha256': sha256_file(args.data_dir / 'dataset_manifest.json'),
                'official_code_revision': subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip(),
                'evaluator_sha256': sha256_file(Path(__file__)),
                'helper_sha256': sha256_file(ROOT / 'scripts/eval_gui_reflection.py'),
                'prompt_template': template, 'requested_pairs': args.limit_pairs,
                'requested_judgments': len(prepared),
                'label_semantics': 'Whether the specified action purpose is fulfilled, not overall task completion.',
                'input_policy': 'Original PNG screenshots, same ordered pair and official action-purpose text; no added fields.',
                'generation': generation, 'batch_size': 1, 'gpu_uuid': args.gpu_uuid,
                'memory_cap_gib': args.memory_cap_gib,
                'versions': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'transformers', 'peft', 'Pillow']}}
    if args.model == 'gui-reflection':
        model_path = ROOT / 'artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain'
        model_source = json.loads(model_path.with_name(model_path.name + '.download.json').read_text())
        manifest.update({'model_repo': model_source['repo_id'], 'model_revision': model_source['revision'],
                         'precision': 'bfloat16', 'preprocessing': 'Official dynamic_preprocess_v3, max_num=6, thumbnails, image_size=448.'})
    else:
        base_root = ROOT / 'artifacts/models/qreflector_base'
        base_source = json.loads((base_root / 'source.json').read_text())
        model_path = base_root / 'qwen3-vl-8b-instruct-unsloth-bnb-4bit'
        adapter_config = json.loads((args.adapter / 'adapter_config.json').read_text())
        assert adapter_config['base_model_name_or_path'] == base_source['repo']
        manifest.update({'base_repo': base_source['repo'], 'base_revision': base_source['revision'],
                         'adapter': str(args.adapter.resolve()),
                         'adapter_sha256': sha256_file(args.adapter / 'adapter_model.safetensors'),
                         'adapter_config_sha256': sha256_file(args.adapter / 'adapter_config.json'),
                         'precision': 'Original prequantized bitsandbytes 4-bit base; bfloat16 compute; saved LoRA.',
                         'preprocessing': 'Saved Qwen3-VL processor applied to original PNGs; no manual downsampling.',
                         'prompt_serialization': 'Official <image> placeholders supplied as native image content entries; remaining text unchanged; truncation disabled.',
                         'loader': 'Transformers + PEFT, without Unsloth runtime patches.'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing output requires identical manifest and --resume')
    else:
        save_json(manifest_path, manifest)
    (args.output_dir / 'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
    predictions = args.output_dir / 'predictions.jsonl'
    results = []
    if predictions.exists():
        if not args.resume:
            raise RuntimeError('Predictions already exist')
        results = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
        for row, expected in zip(results, prepared):
            assert row['sample_id'] == expected['sample_id'] and row['expected'] == expected['expected']
        if len(results) == len(prepared):
            print('Already complete', flush=True)
            return
        if len(results) > len(prepared):
            raise RuntimeError('Saved predictions exceed expected count')
    save_json(args.output_dir / 'gpu_before.json', gpu_snapshot(args.gpu_uuid))
    save_json(args.output_dir / 'status.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(results), 'total': len(prepared)})
    print(f'Loading {args.model} on GPU {args.gpu_uuid}', flush=True)
    if args.model == 'gui-reflection':
        sys.path.insert(0, str(official / 'internvl_chat'))
        from eval.infer import Model
        backend = Model(str(model_path), max_num=6, auto=False)
        tokens = {}
        generate = backend.model.generate

        def capture(*pos, **kw):
            tokens['input_tokens'] = kw['input_ids'].shape[-1]
            output = generate(*pos, **kw)
            tokens['output_tokens'] = output.shape[-1]
            return output

        backend.model.generate = capture

        def infer(prompt, images):
            response = backend(prompt, images, dict(generation))
            return response, dict(tokens)
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors import safe_open
        processor = AutoProcessor.from_pretrained(str(args.adapter), local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16,
                                                         device_map={'': 'cuda:0'}, local_files_only=True)
        backend = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
        loaded_adapter = get_peft_model_state_dict(backend)
        with safe_open(str(args.adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
            assert set(loaded_adapter) == set(saved.keys()), 'Adapter parameter names do not match'
            for name, weight in loaded_adapter.items():
                assert torch.equal(weight.detach().cpu(), saved.get_tensor(name).to(weight.dtype)), name
        save_json(args.output_dir / 'adapter_validation.json', {
            'tensor_count': len(loaded_adapter), 'all_saved_tensors_match_loaded': True,
            'active_adapters': backend.active_adapters})
        del loaded_adapter
        print('Active LoRA:', backend.active_adapters, flush=True)

        def infer(prompt, images):
            if not prompt.startswith('<image>\n<image>\n'):
                raise ValueError('Unexpected official image placeholders')
            text = prompt[len('<image>\n<image>\n'):]
            content = [{'type': 'image', 'image': image} for image in images]
            content.append({'type': 'text', 'text': text})
            inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                  add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors='pt', truncation=False)
            inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
            output = backend.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                      num_beams=1, repetition_penalty=1.0,
                                      temperature=None, top_k=None, top_p=None)
            new_ids = output[0, inputs['input_ids'].shape[-1]:]
            return processor.decode(new_ids, skip_special_tokens=True).strip(), {
                'input_tokens': inputs['input_ids'].shape[-1], 'output_tokens': len(new_ids)}
    print(f'Loaded, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
    started = time.monotonic()
    status = {}
    with predictions.open('a') as output:
        for sample in prepared[len(results):]:
            if torch.cuda.mem_get_info()[0] < 8 * 2**30:
                torch.cuda.empty_cache()
                while torch.cuda.mem_get_info()[0] < 8 * 2**30:
                    print('Waiting for shared GPU memory headroom', flush=True)
                    time.sleep(10)
            images = []
            for name in sample['images']:
                with Image.open(args.data_dir / 'images' / name) as image:
                    images.append(image.convert('RGB'))
            t0 = time.monotonic()
            with torch.inference_mode():
                response, token_counts = infer(sample['prompt'], images)
            torch.cuda.synchronize()
            verdict = parse_verdict(response)
            # Preserve the official startswith scoring separately from strict parsing.
            official_correct = response.strip().lower().startswith(sample['expected'])
            row = {k: sample[k] for k in ['sample_id', 'pair_id', 'kind', 'expected', 'purpose', 'images']}
            row.update({'predicted': verdict, 'correct': verdict == sample['expected'],
                        'official_correct': official_correct, 'full_response': response,
                        'prompt_sha256': hashlib.sha256(sample['prompt'].encode()).hexdigest(),
                        'latency_seconds': time.monotonic() - t0, **token_counts})
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush(); os.fsync(output.fileno())
            results.append(row)
            summary = summarize(results)
            summary['official_prefix_accuracy'] = sum(r['official_correct'] for r in results) / len(results)
            summary['response_counts'] = dict(Counter(r['predicted'] or 'invalid' for r in results))
            save_json(args.output_dir / 'summary.json', summary)
            status = {'status': 'running', 'pid': os.getpid(), 'completed': len(results),
                      'total': len(prepared), 'elapsed_seconds': time.monotonic() - started,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                      'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                      'updated_at_utc': datetime.now(timezone.utc).isoformat()}
            save_json(args.output_dir / 'status.json', status)
            if len(results) <= 4 or len(results) % 50 == 0:
                print(f'{len(results)}/{len(prepared)}; {row["latency_seconds"]:.2f}s; '
                      f'output={row["output_tokens"]} tokens; parsed={verdict is not None}; '
                      f'peak={status["peak_allocated_gib"]:.2f} GiB', flush=True)
    status['status'] = 'complete'
    save_json(args.output_dir / 'status.json', status)
    save_json(args.output_dir / 'gpu_after.json', gpu_snapshot(args.gpu_uuid))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
