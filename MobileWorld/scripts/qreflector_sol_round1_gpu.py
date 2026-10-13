#!/usr/bin/env python3
"""GPU stages for the fixed one-round Sol experiment; never manage other jobs."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import qreflector_sol_round1 as protocol
from qreflector_sol_round1 import (
    ADAPTER, BASE, DATA, GPU, PROCESSOR_BASE, SEED, all_corrections, correction,
    environment, load_plan, read_rows, rows_if_exists, shared, status, target_sample,
)
from qreflector_sft_bf16 import check_bf16_base, gpu_setup, strict_structure
from eval_gui_reflection import parse_verdict, summarize


def setup(run):
    environment(GPU, run / 'hf_cache')
    import torch
    free, _ = torch.cuda.mem_get_info()
    if free < (protocol.CAP_GIB + protocol.RESERVE_GIB) * 1024**3:
        raise RuntimeError('Insufficient GPU headroom before model loading')
    gpu_setup(torch, protocol.CAP_GIB)
    return torch


def load_model(run, adapter, trainable=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    shared.gpu_guard(torch, protocol.CAP_GIB + protocol.RESERVE_GIB)
    processor = AutoProcessor.from_pretrained(str(adapter), local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    precision = check_bf16_base(base)
    base.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE), local_files_only=True)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    count = shared.verify_loaded_adapter(model, adapter, shared.STUDENT)
    if trainable:
        shared.select_adapter(model, shared.STUDENT)
    model.eval()
    return model, processor, {**precision, 'adapter_verified_tensors': count}


def prompt_inputs(processor, row):
    messages = shared.materialize_messages(correction.student_messages(row), DATA, 512)
    encoded = shared.encode_messages(processor, messages)
    return messages, encoded


def generate(run):
    plan, samples = load_plan(run)
    existing = rows_if_exists(run / 'student_rollouts.jsonl')
    done = {r['sample_id'] for r in existing}
    assert len(existing) == len(done)
    by_id = {s['sample_id']: s for s in samples}
    for row in existing:
        assert row['student_messages_sha256'] == by_id[row['sample_id']]['student_messages_sha256']
    if len(done) == 880:
        status(run, 'generate', status='complete', completed=880, total=880)
        return
    torch = setup(run)
    status(run, 'generate', status='loading', completed=len(done), total=880)
    model, processor, precision = load_model(run, ADAPTER)
    shared.write_json(run / 'generation_model_validation.json', precision)
    order = list(range(len(samples)))
    random.Random(SEED).shuffle(order)
    begun = time.monotonic()
    for index in order:
        sample = samples[index]
        if sample['sample_id'] in done:
            continue
        shared.gpu_guard(torch, protocol.RESERVE_GIB)
        torch.manual_seed(SEED + index)
        started = time.monotonic()
        messages, inputs = prompt_inputs(processor, sample['row'])
        inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k,v in inputs.items()}
        n = inputs['input_ids'].shape[-1]
        assert n + 1024 <= 8192
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=1., top_p=1., top_k=0,
                max_new_tokens=1024, num_beams=1, repetition_penalty=1.)
            ids = output[0,n:]
            text = processor.decode(ids, skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        record = {'sample_id': sample['sample_id'], 'train_line': sample['train_line'],
                  'student_messages_sha256': sample['student_messages_sha256'],
                  'full_response': text, 'expected': sample['expected'], 'predicted': parse_verdict(text),
                  'valid_four_field_json': strict_structure(text), 'input_tokens': int(n),
                  'output_tokens': len(ids), 'at_generation_limit': len(ids) >= 1024,
                  'seed': SEED+index, 'latency_seconds': time.monotonic()-started,
                  'provenance': {'imported_same_initial_student': False, 'source_adapter_sha256': plan['source_adapter_sha256']}}
        correction.append_json(run / 'student_rollouts.jsonl', record)
        done.add(sample['sample_id'])
        status(run, 'generate', status='running', completed=len(done), total=880,
               elapsed_seconds=time.monotonic()-begun, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        if len(done) % 20 == 0:
            print(f'Initial student drafts {len(done)}/880', flush=True)
        del output, inputs, ids, messages
    load_plan(run)
    status(run, 'generate', status='complete', completed=880, total=880, elapsed_seconds=time.monotonic()-begun)


def training_rows(run, smoke):
    plan, samples = load_plan(run)
    if smoke:
        by_id = {s['sample_id']: s for s in samples}
        accepted = sorted((r for r in all_corrections(run) if r['accepted']), key=lambda r:r['train_line'])
        assert len(accepted) >= 8
        rows = [target_sample(by_id[r['sample_id']], r) for r in accepted[:8]]
        return plan, rows
    manifest = shared.read_json(run / 'correction_manifest.json')
    assert manifest['status'] == 'frozen'
    assert shared.digest(run / 'corrected_train.jsonl') == manifest['corrected_train_sha256']
    assert shared.digest(run / 'plan.json') == manifest['plan_sha256']
    return plan, read_rows(run / 'corrected_train.jsonl')


def save_checkpoint(run, dest, model, processor, optimizer, state):
    import torch
    assert not dest.exists(), f'Never overwrite a checkpoint: {dest}'
    temporary = dest.parent / ('.saving-' + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate()},
               temporary / 'optimizer_rng.pt')
    shared.write_json(temporary / 'round_state.json', {**state, 'complete': True,
        'plan_sha256': shared.digest(run / 'plan.json'),
        'training_data_sha256': shared.digest(run / 'corrected_train.jsonl')})
    temporary.rename(dest)


def train(run, smoke=False):
    plan, samples = training_rows(run, smoke)
    stage = 'smoke' if smoke else 'train'
    config = {**plan['training'], 'loss_chunk_tokens': 16}
    config['max_sequence_length'] = 8192
    if not smoke and (run / 'train/final').exists():
        state = shared.read_json(run / 'train/final/round_state.json')
        assert state['complete'] and state['examples_seen'] == len(samples)
        assert state['plan_sha256'] == shared.digest(run / 'plan.json')
        status(run, stage, status='complete', resumed_final=True, **{k:state[k] for k in ('step','examples_seen')})
        return
    # Audit all accepted targets and exact unprivileged prefixes before the first update.
    torch = setup(run)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(ADAPTER), local_files_only=True)
    eos_ids = shared.eos_from_processor(processor)
    lengths = []
    for sample in samples:
        messages, prompt = prompt_inputs(processor, sample['row'])
        target = correction.encode_target(processor, messages, prompt, shared.target_of(sample['row']), eos_ids, config)
        lengths.append({'sample_id': sample['sample_id'], 'prompt_tokens': prompt['input_ids'].shape[1],
                        'target_tokens': target.shape[1]})
    shared.write_json(run / f'{stage}_token_audit.json', {'status': 'passed', 'samples': lengths,
        'target_prefix_exact': True, 'student_reference_leakage': False, 'truncated_targets': 0})
    destination = run / 'train'
    destination.mkdir(exist_ok=True)
    previous = sorted((p for p in destination.glob('checkpoint-*') if (p/'round_state.json').exists()),
                      key=lambda p:int(p.name.split('-')[1])) if not smoke else []
    source = previous[-1] if previous else ADAPTER
    status(run, stage, status='loading', source=str(source), examples=len(samples))
    model, processor, precision = load_model(run, source, trainable=True)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 43646976
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    shared.write_json(run / f'{stage}_model_validation.json', {**precision, 'frozen_base': True,
                       'trainable_parameters': sum(p.numel() for p in parameters)})
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.)
    batches = shared.epoch_batches(len(samples), 8, 42, 0)
    # Smoke runs one production-sized batch using the full-run warmup scale.
    total = math.ceil(880/8) if smoke else len(batches)
    state = {'step': 0, 'examples_seen': 0}
    if previous:
        saved = shared.read_json(source / 'round_state.json')
        assert saved['plan_sha256'] == shared.digest(run / 'plan.json')
        assert saved['training_data_sha256'] == shared.digest(run / 'corrected_train.jsonl')
        state = {k:saved[k] for k in state}
        payload = torch.load(source / 'optimizer_rng.pt', weights_only=True, map_location='cpu')
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        random.setstate(payload['python_rng'])
    else:
        torch.manual_seed(42)
        random.seed(42)
    initial_fingerprint = shared.fingerprint_adapter(model, shared.STUDENT) if smoke else None
    begun = time.monotonic()
    metrics = {**state, 'total_steps': total, 'resume_source': str(source),
               'elapsed_seconds': 0., 'resumed_after_last_optimizer_step': state['step'] == len(batches)}
    for bi in range(state['step'], len(batches)):
        batch = batches[bi]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        model.train()
        for index in batch:
            shared.gpu_guard(torch, protocol.RESERVE_GIB)
            row = samples[index]['row']
            messages, prompt = prompt_inputs(processor, row)
            target_ids = correction.encode_target(processor, messages, prompt, shared.target_of(row), eos_ids, config).to('cuda:0')
            gpu_prompt = {k:v.to('cuda:0') for k,v in prompt.items()}
            logits = shared.response_logits(model, gpu_prompt, target_ids)
            loss = shared.reference_cross_entropy(logits, target_ids, chunk_tokens=16)
            assert bool(torch.isfinite(loss)), 'Non-finite loss'
            losses.append(float(loss.detach()))
            (loss/len(batch)).backward()
            del logits, loss, target_ids, gpu_prompt, messages, prompt
        shared.gpu_guard(torch, protocol.RESERVE_GIB)
        assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        lr = shared.learning_rate_at(bi+1, total, config['learning_rate'], config['warmup_ratio'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.update(step=bi+1, examples_seen=state['examples_seen']+len(batch))
        metrics = {**state, 'batch_examples':len(batch), 'loss':sum(losses)/len(losses),
                   'gradient_norm':norm, 'learning_rate':lr, 'total_steps':total,
                   'elapsed_seconds':time.monotonic()-begun,
                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                   'resume_source':str(source)}
        correction.append_json(run / f'{stage}_metrics.jsonl', metrics)
        status(run, stage, status='training', **metrics)
        print(json.dumps(metrics), flush=True)
        if not smoke and ((bi+1)%25==0 or bi+1==len(batches)):
            save_checkpoint(run, destination/f'checkpoint-{bi+1}', model, processor, optimizer, state)
    load_plan(run)
    if smoke:
        changed = shared.fingerprint_adapter(model, shared.STUDENT) != initial_fingerprint
        assert changed and norm > 0
        status(run, stage, status='complete', **metrics, adapter_changed=True,
               source_adapter_unchanged=True, disposable_weights_discarded=True)
    else:
        save_checkpoint(run, destination/'final', model, processor, optimizer, state)
        assert shared.digest(destination/'final/adapter_model.safetensors') != plan['source_adapter_sha256']
        status(run, stage, status='complete', **metrics, source_adapter_unchanged=True,
               final_adapter=str(destination/'final'))


def evaluate(run):
    plan, _ = load_plan(run)
    assert shared.read_json(run/'train_status.json')['status'] == 'complete'
    adapter = run/'train/final'
    final_state = shared.read_json(adapter/'round_state.json')
    assert final_state['plan_sha256'] == shared.digest(run/'plan.json')
    assert final_state['examples_seen'] == shared.read_json(run/'correction_manifest.json')['accepted']
    assert shared.digest(DATA/'test.jsonl') == plan['test_sha256']
    rows, labels = read_rows(DATA/'test.jsonl'), read_rows(DATA/'test.labels.jsonl')
    assert len(rows)==len(labels)==1082
    destination = run/'test'
    destination.mkdir(exist_ok=True)
    manifest = {'data':str(DATA),'data_sha256':plan['test_sha256'],'adapter':str(adapter),
                'adapter_sha256':shared.digest(adapter/'adapter_model.safetensors'),
                'base':str(BASE),'dtype':'bfloat16','quantization':None,
                'image_max_side':512,'max_new_tokens':1024,'do_sample':False,'batch_size':1,
                'prompt_template_sha256':shared.digest(DATA/'prompt_template.txt'),
                'checkpoint_policy':plan['evaluation']['checkpoint_policy'],'validation_used':False,
                'gpu_uuid':GPU,'memory_cap_gib':32,'processor_source':str(ADAPTER),
                'requested_samples':1082,'evaluator_sha256':shared.digest(__file__)}
    if (destination/'manifest.json').exists():
        assert shared.read_json(destination/'manifest.json')==manifest
    else:
        shared.write_json(destination/'manifest.json',manifest)
    results = rows_if_exists(destination/'predictions.jsonl')
    for i,r in enumerate(results):
        assert r['sample_id']==i+1 and r['source_sample_id']==labels[i]['sample_id']
        assert r['expected']==labels[i]['answer'].lower()
    if len(results)==1082:
        shared.write_json(destination/'summary.json',summarize(results))
        status(run,'test',status='complete',completed=1082,total=1082)
        return
    torch=setup(run)
    status(run,'test',status='loading',completed=len(results),total=1082)
    model,processor,precision=load_model(run,adapter)
    shared.write_json(destination/'model_validation.json',precision)
    begun=time.monotonic()
    for i in range(len(results),1082):
        shared.gpu_guard(torch,protocol.RESERVE_GIB)
        started=time.monotonic()
        messages,inputs=prompt_inputs(processor,rows[i])
        prompt_text='\n'.join(c['text'] for c in messages[0]['content'] if c['type']=='text')
        inputs={k:v.to('cuda:0') for k,v in inputs.items()}
        n=inputs['input_ids'].shape[-1]
        assert n+1024<=8192
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=1024,do_sample=False,num_beams=1,
                temperature=None,top_p=None,top_k=None,repetition_penalty=1.)
            ids=generated[0,n:]
            response=processor.decode(ids,skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        predicted=parse_verdict(response)
        expected=labels[i]['answer'].lower()
        record={'sample_id':i+1,'source_sample_id':labels[i]['sample_id'],'expected':expected,
                'predicted':predicted,'correct':predicted==expected,'full_response':response,
                'valid_four_field_json':strict_structure(response),
                'prompt_sha256':hashlib.sha256(prompt_text.encode()).hexdigest(),
                'input_tokens':int(n),'output_tokens':len(ids),'at_generation_limit':len(ids)>=1024,
                'latency_seconds':time.monotonic()-started}
        correction.append_json(destination/'predictions.jsonl',record)
        results.append(record)
        shared.write_json(destination/'summary.json',summarize(results))
        status(run,'test',status='running',completed=len(results),total=1082,
               elapsed_seconds=time.monotonic()-begun)
        if len(results)%25==0:
            print(f'Test {len(results)}/1082; accuracy={summarize(results)["accuracy"]:.4f}',flush=True)
        del inputs,generated,ids,messages
    status(run,'test',status='complete',completed=1082,total=1082,elapsed_seconds=time.monotonic()-begun)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','smoke','train','test'])
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve()
    with (run/'.gpu_stage.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.stage=='smoke':train(run,smoke=True)
            elif args.stage=='test':evaluate(run)
            else:globals()[args.stage](run)
        except Exception as error:
            status(run,args.stage,status='failed',error_type=type(error).__name__,error=str(error))
            if isinstance(error,RuntimeError) and ('headroom' in str(error) or 'GiB free' in str(error)):
                raise SystemExit(75) from None
            raise


if __name__=='__main__':
    main()
