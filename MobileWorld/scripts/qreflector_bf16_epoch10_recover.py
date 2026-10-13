#!/usr/bin/env python3
"""Recover the interrupted epoch-10 run with native Trainer checkpoint resume.

Restore epoch-9 weights, optimizer, existing scheduler and RNG. Do not restart LR.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from qreflector_sft_bf16 import (BASE, DATA, PROCESSOR_BASE, ROOT, check_bf16_base,
    digest, environment, gpu_setup, read_rows, validate_dataset, vlm_sample)
from eval_gui_reflection import gpu_snapshot, save_json

GPU_UUID = 'GPU-991ac45f-e9e9-1c25-590c-fb49ca752965'
SOURCE = ROOT / 'artifacts/training/qreflector_bf16_v6_continue_20261005_gpu0/training/checkpoint-222'


def verify_optimizer_state(saved, loaded):
    """Check restoration of moments/counters, parameter mapping and group values."""
    import torch
    def compare(value, actual, path):
        if torch.is_tensor(value):
            assert torch.is_tensor(actual) and value.dtype == actual.dtype, path
            assert torch.equal(value.cpu(), actual.cpu()), path
            return 1
        if isinstance(value, dict):
            assert isinstance(actual, dict) and value.keys() == actual.keys(), path
            return sum(compare(v, actual[k], f'{path}.{k}') for k, v in value.items())
        if isinstance(value, (list, tuple)):
            assert type(actual) is type(value) and len(value) == len(actual), path
            return sum(compare(v, a, f'{path}.{i}') for i, (v, a) in enumerate(zip(value, actual)))
        assert value == actual, path
        return 1
    assert saved['state']
    return compare(saved, loaded, 'optimizer')


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
    # Initialize the same adapter architecture, then restore ALL saved LoRA tensors.
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    recovery = args.resume.resolve()
    assert recovery.parent == args.output.resolve()
    source_state = json.loads((recovery / 'trainer_state.json').read_text())
    assert source_state['global_step'] == 444 and source_state['epoch'] == 4.0
    saved = load_file(str(recovery / 'adapter_model.safetensors'), device='cpu')
    loaded = get_peft_model_state_dict(model)
    assert set(saved) == set(loaded), 'Source adapter scope differs'
    for name, weight in loaded.items():
        assert weight.shape == saved[name].shape, name
    result = set_peft_model_state_dict(model, saved)
    assert not result.unexpected_keys, result.unexpected_keys
    loaded = get_peft_model_state_dict(model)
    for name, weight in loaded.items():
        assert torch.equal(weight.detach().cpu(), saved[name].to(weight.dtype)), name
    save_json(args.output / 'recovery_adapter_validation.json', {
        'source': str(recovery), 'source_sha256': digest(recovery / 'adapter_model.safetensors'),
        'all_saved_tensors_match_loaded': True, 'tensor_count': len(saved),
        'source_step': 999, 'source_epochs': 9, 'optimizer_reset': False,
        'scheduler_reset': False, 'exact_training_resume': True,
    })
    del loaded, saved, result
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
        'purpose': 'disposable one-step smoke test' if args.smoke_test else 'five additional epochs from cumulative BF16 step 555; restored optimizer and restarted schedule',
        'precision': 'unquantized BF16 base; BF16 mixed-precision LoRA training',
        'processor_source': str(PROCESSOR_BASE),
        'model_source': json.loads((BASE.parent / 'source.json').read_text()),
        'train_sha256': digest(args.data / 'train.jsonl'), 'val_sha256': digest(args.data / 'val.jsonl'),
        'epochs': 5, 'source_step': 555, 'cumulative_final_step': 1110,
        'source_adapter': str(SOURCE), 'optimizer_reset': False, 'scheduler_reset': True, 'batch_size': 1, 'gradient_accumulation_steps': 8,
        'expected_optimizer_steps': 5 * math.ceil(882 / 8),
        'learning_rate': 2e-5, 'warmup_ratio': 0.05, 'lora_rank': 16, 'lora_alpha': 16,
        'trainable_parameters': trainable, 'image_max_side': args.image_max_side,
        'max_sequence_length': 8192, 'assistant_only_loss': True,
        'save_steps': 111, 'save_total_limit': None, 'load_best_model_at_end': False,
        'seed': 42, 'data_seed': 47, 'memory_cap_gib': args.memory_cap_gib,
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'trl', 'unsloth', 'unsloth_zoo', 'peft']}}
    params.update(recovery_checkpoint=str(recovery), recovery_local_step=444, recovery_cumulative_step=999, recovery_scheduler_reset=False)
    save_json(args.output / 'recovery_training_manifest.json', params)
    begun = time.monotonic()

    class Progress(TrainerCallback):
        def on_step_end(self, _args, state, control, **kw):
            save_json(args.output / 'status.json', {
                'status': 'training', 'pid': os.getpid(), 'step': state.global_step,
                'total_steps': state.max_steps, 'epoch': state.epoch,
                'cumulative_step': 555 + state.global_step, 'cumulative_epoch': 5 + state.epoch,
                'elapsed_seconds': time.monotonic() - begun,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            })
            return control

        def on_step_begin(self, _args, state, control, **kw):
            if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                raise RuntimeError('Shared GPU has less than 8 GiB headroom; stopping only this child')
            return control

        def on_log(self, _args, state, control, logs=None, **kw):
            print(json.dumps({'step': state.global_step, **(logs or {})}), flush=True)

    config = SFTConfig(
        output_dir=str(args.output), num_train_epochs=5,
        max_steps=1 if args.smoke_test else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-5,
        lr_scheduler_type='cosine', warmup_ratio=0.05, bf16=True,
        optim='adamw_8bit', seed=42, data_seed=47,
        logging_steps=1 if args.smoke_test else 5,
        save_strategy='no' if args.smoke_test else 'steps', save_steps=111, save_total_limit=None,
        eval_strategy='no' if args.smoke_test else 'steps', eval_steps=111, load_best_model_at_end=False,
        report_to='none', remove_unused_columns=False,
        dataset_kwargs={'skip_prepare_dataset': True}, dataset_text_field='',
        max_length=None, packing=False, dataloader_num_workers=0,
    )
    class OptimizerContinuationTrainer(SFTTrainer):
        def _load_rng_state(self, checkpoint):
            super()._load_rng_state(checkpoint)
            save_json(args.output / 'recovery_rng_validation.json', {
                'checkpoint': str(checkpoint), 'native_trainer_rng_restore_completed': True,
                'rng_state_sha256': digest(Path(checkpoint) / 'rng_state.pth'),
            })

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            saved_optimizer = torch.load(Path(checkpoint) / 'optimizer.pt', map_location='cpu', weights_only=False)
            checked = verify_optimizer_state(saved_optimizer, self.optimizer.state_dict())
            saved_scheduler = torch.load(Path(checkpoint) / 'scheduler.pt', map_location='cpu', weights_only=False)
            assert self.lr_scheduler.state_dict() == saved_scheduler, 'Scheduler was not restored exactly'
            save_json(args.output / 'recovery_optimizer_validation.json', {
                'source': str(checkpoint), 'optimizer_state_exact': True,
                'checked_state_values': checked, 'scheduler_state_exact': True,
                'scheduler_last_epoch': saved_scheduler['last_epoch'],
                'learning_rates': [g['lr'] for g in self.optimizer.param_groups],
            })

    trainer = OptimizerContinuationTrainer(model=model, processing_class=processor,
        train_dataset=training, eval_dataset=validation,
        data_collator=collator, args=config, callbacks=[Progress()])
    result = trainer.train(resume_from_checkpoint=str(recovery))
    if args.smoke_test:
        assert trainer.state.global_step == 1
        assert math.isfinite(result.metrics['train_loss'])
        save_json(args.output / 'recovery_training_metrics.json', result.metrics)
        save_json(args.output / 'status.json', {'status': 'smoke_complete', 'pid': os.getpid(),
                  'step': 1, 'weights_discarded': True,
                  'elapsed_seconds': time.monotonic() - begun,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
        return
    assert trainer.state.global_step == 555, trainer.state.global_step
    assert digest(SOURCE / 'adapter_model.safetensors') != digest(args.output / 'checkpoint-555/adapter_model.safetensors'), 'Adapter did not update'
    final = args.output / 'final'
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    trainer.state.save_to_json(str(final / 'trainer_state.json'))
    save_json(args.output / 'recovery_training_metrics.json', result.metrics)
    for epoch in range(6, 11):
        local_step = (epoch - 5) * 111
        (args.output / f'epoch-{epoch}-step-{epoch * 111}').symlink_to(f'checkpoint-{local_step}', target_is_directory=True)
    for name in [f'checkpoint-{i * 111}' for i in range(1, 6)] + ['final']:
        assert (args.output / name / 'adapter_model.safetensors').is_file(), name
    save_json(args.output / 'status.json', {
        'status': 'complete', 'pid': os.getpid(), 'step': trainer.state.global_step,
        'cumulative_step': 555 + trainer.state.global_step, 'cumulative_epochs': 10,
        'elapsed_seconds': time.monotonic() - begun,
        'checkpoints': [f'checkpoint-{i * 111}' for i in range(1, 6)] + ['final'],
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
    })
    save_json(args.output / 'gpu_after.json', gpu_snapshot(args.gpu_uuid))



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--gpu-uuid', default=GPU_UUID)
    p.add_argument('--memory-cap-gib', type=float, default=40)
    p.add_argument('--image-max-side', type=int, default=512)
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--resume', type=Path, required=True)
    a = p.parse_args()
    train(a)


if __name__ == '__main__':
    main()
