#!/usr/bin/env python3
"""BF16 LoRA SFT and evaluation, matched to the completed v6 QLoRA run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from eval_gui_reflection import ROOT, gpu_snapshot, parse_verdict, save_json, summarize

DATA = ROOT / '../datasets/condition_generation_v6_user_reviewed'
BASE = ROOT / 'artifacts/models/qreflector_base_bf16/Qwen3-VL-8B-Instruct'
PROCESSOR_BASE = ROOT / 'artifacts/models/qreflector_base/qwen3-vl-8b-instruct-unsloth-bnb-4bit'
GPU_UUID = 'GPU-36c832f8-4901-c89c-3c40-8d8b9825661d'
FIELDS = {'precondition', 'postcondition', 'reason', 'answer'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_dataset(data, images=False):
    frozen = json.loads((data / 'FROZEN.json').read_text())
    assert frozen['status'] == 'frozen'
    for name, expected in frozen['artifact_sha256'].items():
        assert digest(data / name) == expected, f'Frozen artifact changed: {name}'
    for split, n in [('train', 882), ('val', 71), ('test', 1082)]:
        rows = read_rows(data / f'{split}.jsonl')
        assert len(rows) == n
        for i, row in enumerate(rows):
            assert [m['role'] for m in row['messages']] == ['user', 'assistant']
            target = json.loads(row['messages'][1]['content'][0]['text'])
            assert set(target) == FIELDS and target['answer'] in ('Yes', 'No')
            paths = [c['image'] for c in row['messages'][0]['content'] if c['type'] == 'image']
            assert len(paths) == 2 and all((data / p).is_file() for p in paths), (split, i)
    if images:
        for name, item in json.loads((data / frozen['image_manifest']).read_text()).items():
            assert digest(data / name) == item['sha256'], name
    return frozen


def environment(gpu_uuid, cache):
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': gpu_uuid, 'HF_HOME': str(cache),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'USE_TF': '0', 'USE_FLAX': '0', 'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    })
    os.nice(10)


def gpu_setup(torch, memory_cap_gib):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU must be visible')
    free, capacity = torch.cuda.mem_get_info(0)
    if free < (memory_cap_gib + 8) * 1024**3:
        raise RuntimeError(f'Insufficient headroom: {free / 1024**3:.1f} GiB free')
    torch.cuda.set_per_process_memory_fraction(memory_cap_gib * 1024**3 / capacity, 0)
    torch.manual_seed(42)


def check_bf16_base(model):
    assert not getattr(model, 'is_loaded_in_4bit', False)
    # Unsloth vision.py sets is_loaded_in_8bit=True even for 16-bit LoRA
    # as a DDP compatibility flag. Inspect actual modules and tensors instead.
    assert not getattr(model.config, 'quantization_config', None)
    assert not any(type(module).__name__ in ('Linear4bit', 'Linear8bitLt')
                   for module in model.modules())
    import torch
    dense = [(name, p) for name, p in model.named_parameters()
             if p.ndim == 2 and '.layers.' in name and 'lora_' not in name]
    assert dense and all(p.dtype == torch.bfloat16 for _, p in dense)
    return {'quantized': False, 'language_matrix_dtype': 'bfloat16',
            'library_8bit_compatibility_flag': bool(getattr(model, 'is_loaded_in_8bit', False)),
            'checked_language_matrices': len(dense)}


def load_images(row, data, max_side=512):
    from PIL import Image
    images = []
    for item in row['messages'][0]['content']:
        if item['type'] == 'image':
            with Image.open(data / item['image']) as opened:
                image = opened.convert('RGB')
            if max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            images.append(image)
    if len(images) != 2:
        raise ValueError('A sample must contain both before and after images')
    return images


def vlm_sample(row, data, max_side):
    images = load_images(row, data, max_side)
    messages = copy.deepcopy(row['messages'])
    for message in messages:
        for item in message['content']:
            if item['type'] == 'image':
                item.clear()
                item['type'] = 'image'
    return {'messages': messages, 'images': images}


def strict_structure(text):
    if text.strip().startswith('```'):
        return False
    try:
        obj = json.loads(text)
        return (isinstance(obj, dict) and set(obj) == FIELDS
                and all(isinstance(v, str) and v.strip() for v in obj.values())
                and obj['answer'] in ('Yes', 'No'))
    except (ValueError, TypeError):
        return False


def train(args):
    validate_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'final').exists():
        raise RuntimeError('Final output already exists; refusing to overwrite')
    environment(args.gpu_uuid, args.output / 'hf_cache')
    os.environ['UNSLOTH_COMPILE_LOCATION'] = str(args.output / 'unsloth_compiled_cache')
    # Unsloth must be imported before Transformers. GPU visibility is already fixed.
    from unsloth import FastModel
    import torch
    from transformers import AutoProcessor, GenerationConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    gpu_setup(torch, args.memory_cap_gib)
    save_json(args.output / 'gpu_before.json', gpu_snapshot(args.gpu_uuid))
    save_json(args.output / 'status.json', {'status': 'loading', 'pid': os.getpid()})
    model, processor = FastModel.from_pretrained(
        model_name=str(BASE), max_seq_length=8192, load_in_4bit=False,
        load_in_8bit=False, full_finetuning=False, dtype=torch.bfloat16,
        device_map={'': 0}, local_files_only=True,
    )
    save_json(args.output / 'precision_validation.json', check_bf16_base(model))
    # Keep tokenization, chat template, image preprocessing and generation
    # defaults identical to the prior QLoRA experiment. Only weights change.
    processor = AutoProcessor.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    model.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    model = FastModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth', random_state=42,
    )
    save_json(args.output / 'precision_after_lora.json', check_bf16_base(model))
    assert all('lora_' in name for name, p in model.named_parameters() if p.requires_grad)
    model.config.use_cache = False
    training = [vlm_sample(row, args.data, args.image_max_side) for row in read_rows(args.data / 'train.jsonl')]
    validation = [vlm_sample(row, args.data, args.image_max_side) for row in read_rows(args.data / 'val.jsonl')]
    collator = UnslothVisionDataCollator(
        model=model, processor=processor, max_seq_length=8192, resize='max',
        train_on_responses_only=True, instruction_part='<|im_start|>user\n',
        response_part='<|im_start|>assistant\n', num_proc=1,
    )
    # Audit every train/val sample before the first optimizer update. No test
    # responses or labels are exposed to the trainer or checkpoint selection.
    audits = []
    for split, samples in [('train', training), ('val', validation)]:
        for i, sample in enumerate(samples, 1):
            batch = collator([sample])
            ids, labels = batch['input_ids'][0], batch['labels'][0]
            length = int(ids.shape[0])
            assert length < 8192, f'Potential truncation in {split}-{i}'
            visible = labels != -100
            assert bool(visible.any()) and int(visible.sum()) < length
            # The loss must cover the complete target, including its final verdict.
            supervised = processor.decode(ids[visible], skip_special_tokens=True).strip()
            expected = sample['messages'][1]['content'][0]['text']
            assert supervised == expected, (split, i, supervised, expected)
            audits.append({'split': split, 'line': i, 'tokens': length,
                           'supervised_tokens': int(visible.sum()), 'target_exact': True})
        print(f'Collator audit: {split} {len(samples)} samples passed', flush=True)
    save_json(args.output / 'collator_audit.json', {
        'assistant_targets_exact': True, 'samples': audits,
        'max_tokens': max(x['tokens'] for x in audits), 'truncated_samples': 0,
    })
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 43646976, f'LoRA parameter scope changed: {trainable}'
    params = {'base_model': str(BASE), 'data': str(args.data),
        'purpose': 'disposable one-step smoke test' if args.smoke_test else 'matched BF16 LoRA experiment',
        'precision': 'unquantized BF16 base; BF16 mixed-precision LoRA training',
        'processor_source': str(PROCESSOR_BASE),
        'model_source': json.loads((BASE.parent / 'source.json').read_text()),
        'train_sha256': digest(args.data / 'train.jsonl'), 'val_sha256': digest(args.data / 'val.jsonl'),
        'epochs': 3, 'batch_size': 1, 'gradient_accumulation_steps': 8,
        'expected_optimizer_steps': 3 * math.ceil(882 / 8),
        'learning_rate': 2e-4, 'lora_rank': 16, 'lora_alpha': 16,
        'trainable_parameters': trainable, 'image_max_side': args.image_max_side,
        'max_sequence_length': 8192, 'assistant_only_loss': True,
        'save_steps': 100, 'save_total_limit': None, 'load_best_model_at_end': False,
        'seed': 42, 'memory_cap_gib': args.memory_cap_gib,
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'trl', 'unsloth', 'unsloth_zoo', 'peft']}}
    save_json(args.output / 'training_manifest.json', params)
    begun = time.monotonic()

    class Progress(TrainerCallback):
        def on_step_end(self, _args, state, control, **kw):
            # Preserve 100/200/300 and manually save final; avoid a duplicate
            # checkpoint-333 and automatic deletion of checkpoint-100.
            if state.global_step == state.max_steps and state.global_step % 100:
                control.should_save = False
            save_json(args.output / 'status.json', {
                'status': 'training', 'pid': os.getpid(), 'step': state.global_step,
                'total_steps': state.max_steps, 'epoch': state.epoch,
                'elapsed_seconds': time.monotonic() - begun,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            })
            return control

        def on_log(self, _args, state, control, logs=None, **kw):
            print(json.dumps({'step': state.global_step, **(logs or {})}), flush=True)

    config = SFTConfig(
        output_dir=str(args.output), num_train_epochs=3,
        max_steps=1 if args.smoke_test else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        lr_scheduler_type='cosine', warmup_ratio=0.1, bf16=True,
        optim='adamw_8bit', seed=42, data_seed=42,
        logging_steps=1 if args.smoke_test else 5,
        save_strategy='no' if args.smoke_test else 'steps', save_steps=100, save_total_limit=None,
        eval_strategy='no' if args.smoke_test else 'steps', eval_steps=100, load_best_model_at_end=False,
        report_to='none', remove_unused_columns=False,
        dataset_kwargs={'skip_prepare_dataset': True}, dataset_text_field='',
        max_length=None, packing=False, dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, processing_class=processor,
        train_dataset=training, eval_dataset=validation,
        data_collator=collator, args=config, callbacks=[Progress()])
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    if args.smoke_test:
        assert trainer.state.global_step == 1
        assert math.isfinite(result.metrics['train_loss'])
        save_json(args.output / 'training_metrics.json', result.metrics)
        save_json(args.output / 'status.json', {'status': 'smoke_complete', 'pid': os.getpid(),
                  'step': 1, 'weights_discarded': True,
                  'elapsed_seconds': time.monotonic() - begun,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
        return
    assert trainer.state.global_step == 333, trainer.state.global_step
    final = args.output / 'final'
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    trainer.state.save_to_json(str(final / 'trainer_state.json'))
    save_json(args.output / 'training_metrics.json', result.metrics)
    for name in ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final']:
        assert (args.output / name / 'adapter_model.safetensors').is_file(), name
    save_json(args.output / 'status.json', {
        'status': 'complete', 'pid': os.getpid(), 'step': trainer.state.global_step,
        'elapsed_seconds': time.monotonic() - begun,
        'checkpoints': ['checkpoint-100', 'checkpoint-200', 'checkpoint-300', 'final'],
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
    })
    save_json(args.output / 'gpu_after.json', gpu_snapshot(args.gpu_uuid))


def evaluate(args):
    validate_dataset(args.data)
    environment(args.gpu_uuid, args.output / 'hf_cache')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    gpu_setup(torch, args.memory_cap_gib)
    rows = read_rows(args.data / 'test.jsonl')
    args.output.mkdir(parents=True, exist_ok=True)
    assert 1 <= args.limit <= 1082
    manifest = {
        'data': str(args.data), 'data_sha256': digest(args.data / 'test.jsonl'),
        'base': str(BASE), 'adapter': str(args.adapter) if args.adapter else None,
        'adapter_sha256': digest(args.adapter / 'adapter_model.safetensors') if args.adapter else None,
        'evaluator_sha256': digest(Path(__file__)),
        'image_max_side': args.image_max_side, 'max_new_tokens': args.max_new_tokens,
        'do_sample': False, 'batch_size': 1, 'quantization': None, 'dtype': 'bfloat16',
        'processor_source': str(PROCESSOR_BASE),
        'model_source': json.loads((BASE.parent / 'source.json').read_text()),
        'prompt': 'Frozen v6 user message only; matched to the completed QLoRA evaluations.',
        'requested_samples': args.limit, 'gpu_uuid': args.gpu_uuid,
        'memory_cap_gib': args.memory_cap_gib,
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft']},
    }
    mf = args.output / 'manifest.json'
    if mf.exists():
        assert args.resume_results and json.loads(mf.read_text()) == manifest, 'Resume manifest mismatch'
    else:
        save_json(mf, manifest)
    predictions = args.output / 'predictions.jsonl'
    results = read_rows(predictions) if predictions.exists() else []
    if results:
        assert args.resume_results
        for i, record in enumerate(results):
            assert record['sample_id'] == i + 1
            assert record['expected'] == json.loads(rows[i]['messages'][1]['content'][0]['text'])['answer'].lower()
        if len(results) == args.limit:
            return
    save_json(args.output / 'gpu_before.json', gpu_snapshot(args.gpu_uuid))
    save_json(args.output / 'status.json', {'status': 'loading', 'pid': os.getpid(), 'completed': len(results), 'total': args.limit})
    processor = AutoProcessor.from_pretrained(str(args.adapter or PROCESSOR_BASE), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    model.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    save_json(args.output / 'precision_validation.json', check_bf16_base(model))
    if args.adapter:
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors import safe_open
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        loaded = get_peft_model_state_dict(model)
        with safe_open(str(args.adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
            assert set(loaded) == set(saved.keys())
            for name, weight in loaded.items():
                assert torch.equal(weight.detach().cpu(), saved.get_tensor(name).to(weight.dtype)), name
        save_json(args.output / 'adapter_validation.json', {'all_saved_tensors_match_loaded': True, 'tensor_count': len(loaded)})
        del loaded
    else:
        assert not hasattr(model, 'peft_config'), 'Base baseline must not have a LoRA adapter'
    model.eval()
    begun = time.monotonic()
    with predictions.open('a') as output:
        for i in range(len(results), args.limit):
            while torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                torch.cuda.empty_cache()
                print('Waiting for shared GPU headroom; no other process is changed', flush=True)
                time.sleep(10)
            row = rows[i]
            images = iter(load_images(row, args.data, args.image_max_side))
            user = copy.deepcopy(row['messages'][0])
            for item in user['content']:
                if item['type'] == 'image':
                    item['image'] = next(images)
            prompt_text = '\n'.join(c['text'] for c in user['content'] if c['type'] == 'text')
            started = time.monotonic()
            with torch.inference_mode():
                inputs = processor.apply_chat_template([user], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors='pt', truncation=False)
                inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                    repetition_penalty=1.0)
                new_ids = output_ids[0, inputs['input_ids'].shape[-1]:]
                response = processor.decode(new_ids, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            expected = json.loads(row['messages'][1]['content'][0]['text'])['answer'].lower()
            predicted = parse_verdict(response)
            record = {'sample_id': i + 1, 'expected': expected, 'predicted': predicted,
                'correct': predicted == expected, 'full_response': response,
                'valid_four_field_json': strict_structure(response),
                'prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
                'input_tokens': int(inputs['input_ids'].shape[-1]), 'output_tokens': len(new_ids),
                'at_generation_limit': len(new_ids) >= args.max_new_tokens,
                'latency_seconds': time.monotonic() - started}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
            results.append(record)
            summary = summarize(results)
            summary['valid_four_field_json_rate'] = sum(r['valid_four_field_json'] for r in results) / len(results)
            save_json(args.output / 'summary.json', summary)
            save_json(args.output / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'completed': len(results), 'total': args.limit, 'elapsed_seconds': time.monotonic() - begun,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
            if len(results) <= 3 or len(results) % 25 == 0:
                print(f'{len(results)}/{args.limit} accuracy={summary["accuracy"]:.4f}; output={len(new_ids)} tokens; {record["latency_seconds"]:.2f}s', flush=True)
    save_json(args.output / 'status.json', {'status': 'complete', 'pid': os.getpid(),
        'completed': len(results), 'total': args.limit, 'elapsed_seconds': time.monotonic() - begun})
    save_json(args.output / 'gpu_after.json', gpu_snapshot(args.gpu_uuid))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate', 'validate'])
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--gpu-uuid', default=GPU_UUID)
    p.add_argument('--memory-cap-gib', type=float, default=40)
    p.add_argument('--image-max-side', type=int, default=512)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--limit', type=int, default=1082)
    p.add_argument('--resume', type=Path)
    p.add_argument('--resume-results', action='store_true')
    p.add_argument('--smoke-test', action='store_true', help='One disposable optimizer step; no weights saved')
    a = p.parse_args()
    if a.mode == 'validate':
        print(json.dumps(validate_dataset(a.data, images=True)['splits'], indent=2))
    elif a.mode == 'train':
        train(a)
    else:
        evaluate(a)


if __name__ == '__main__':
    main()
