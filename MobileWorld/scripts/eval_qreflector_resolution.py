#!/usr/bin/env python3
"""Fixed-checkpoint validation-only resolution comparison. Never manages other jobs."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from qreflector_sft_bf16 import BASE, PROCESSOR_BASE, ROOT, check_bf16_base, digest, environment, gpu_setup, load_images, read_rows, strict_structure
from eval_gui_reflection import parse_verdict, summarize, save_json
import qreflector_geometry_prompt as geometry
import qreflector_opsd as shared

DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
ADAPTER = ROOT / 'artifacts/training/qreflector_sol_round2_geometry_20261009_gpu2/train/final'
PROCESSOR = ROOT / 'artifacts/training/qreflector_sol_round1_v9_epoch9_20261008_gpu7/train/final'


def gpu_state(uuid):
    return subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True).strip()


def evaluate(args):
    run = args.output.resolve()
    assert ROOT / 'artifacts/evaluations' in run.parents
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            evaluate_locked(args, run)
        except BaseException as exc:
            save_json(run / 'status.json', {'status':'failed', 'pid':os.getpid(), 'error':f'{type(exc).__name__}: {exc}'})
            raise


def evaluate_locked(args, run):
    assert not (run / 'predictions.jsonl').exists(), 'Use a fresh output directory; no implicit overwrite.'
    environment(args.gpu_uuid, run / 'hf_cache')
    occupied = gpu_state(args.gpu_uuid)
    if occupied:
        raise RuntimeError(f'Refusing to load on an occupied GPU: {occupied}')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
    gpu_setup(torch, 32)
    rows = read_rows(DATA / 'val.jsonl')
    labels = read_rows(DATA / 'val.labels.jsonl')
    assert len(rows) == len(labels) == 71
    files = [DATA/'val.jsonl', DATA/'val.labels.jsonl', DATA/'prompt_template.txt',
             ADAPTER/'adapter_model.safetensors', ADAPTER/'adapter_config.json',
             PROCESSOR/'preprocessor_config.json', PROCESSOR/'chat_template.jinja',
             Path(geometry.__file__), Path(__file__)]
    hashes = {str(p):digest(p) for p in files}
    manifest = {'split':'val', 'rows':71, 'adapter':str(ADAPTER), 'processor':str(PROCESSOR),
        'base':str(BASE), 'image_max_side':args.max_side, 'gpu_uuid':args.gpu_uuid,
        'dtype':'bfloat16', 'quantization':None, 'do_sample':False, 'seed':42,
        'max_new_tokens':1024, 'batch_size':1, 'memory_cap_gib':32,
        'prompt':'Frozen v9 with dynamic geometry, consistent with round-2 inference.',
        'comparison':'Only input image resolution and its geometry dimensions change.',
        'checkpoint_selection':'Latest completed round-2 checkpoint fixed before validation; no test-score selection.',
        'pinned_files':hashes, 'pid':os.getpid(), 'started_at_unix':time.time(),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft']}}
    save_json(run/'manifest.json',manifest)
    save_json(run/'status.json',{'status':'loading','completed':0,'total':71,'pid':os.getpid()})
    processor = AutoProcessor.from_pretrained(str(PROCESSOR),local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(str(BASE),dtype=torch.bfloat16,
        device_map={'':'cuda:0'},local_files_only=True,attn_implementation='sdpa')
    base.generation_config = GenerationConfig.from_pretrained(str(PROCESSOR_BASE),local_files_only=True)
    precision = check_bf16_base(base)
    model = PeftModel.from_pretrained(base,str(ADAPTER),is_trainable=False)
    precision['adapter_verified_tensors'] = shared.verify_loaded_adapter(model,ADAPTER,shared.STUDENT)
    assert not any(p.requires_grad for p in model.parameters())
    save_json(run/'model_validation.json',precision)
    model.eval()
    results = []
    start = time.monotonic()
    for i,row in enumerate(rows):
        if torch.cuda.mem_get_info()[0] < 8*1024**3:
            raise RuntimeError('Insufficient free GPU headroom; stopping this worker without signaling any process.')
        augmented = geometry.augment_row(row,DATA,args.max_side)
        user = copy.deepcopy(augmented['messages'][0])
        assert user['role']=='user'
        text = user['content'][0]['text']
        images = load_images(row,DATA,args.max_side)
        geom = geometry.geometry(row,DATA,args.max_side)
        assert [list(im.size) for im in images] == [[g['input_width'],g['input_height']] for g in geom['images']]
        it = iter(images)
        for part in user['content']:
            if part['type']=='image': part['image']=next(it)
        begun = time.monotonic()
        inputs = shared.encode_messages(processor,[user])
        grid = inputs['image_grid_thw'].tolist()
        n = inputs['input_ids'].shape[-1]
        assert n+1024<=8192
        inputs = {k:v.to('cuda:0') for k,v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs,max_new_tokens=1024,do_sample=False,num_beams=1,
                temperature=None,top_p=None,top_k=None,repetition_penalty=1.)
            ids = output[0,n:]
            response = processor.decode(ids,skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        pred = parse_verdict(response)
        expected = labels[i]['answer'].lower()
        assert expected == json.loads(row['messages'][1]['content'][0]['text'])['answer'].lower()
        rec = {'sample_id':i+1,'source_sample_id':labels[i]['sample_id'],
            'expected':expected,'predicted':pred,'correct':pred==expected,'full_response':response,
            'valid_four_field_json':strict_structure(response),
            'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'context_sha256':hashlib.sha256(json.dumps(geometry.context_of(row),sort_keys=True).encode()).hexdigest(),
            'image_geometry':geom,'image_grid_thw':grid,
            'processor_image_sizes_wh':[[g[2]*processor.image_processor.patch_size,g[1]*processor.image_processor.patch_size] for g in grid],
            'input_tokens':int(n),'output_tokens':len(ids),'at_generation_limit':len(ids)>=1024,
            'latency_seconds':time.monotonic()-begun}
        with (run/'predictions.jsonl').open('a') as handle:
            handle.write(json.dumps(rec,ensure_ascii=False)+'\n'); handle.flush(); os.fsync(handle.fileno())
        results.append(rec)
        save_json(run/'summary.json',summarize(results))
        save_json(run/'status.json',{'status':'running','completed':len(results),'total':71,
            'pid':os.getpid(),'elapsed_seconds':time.monotonic()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
        if len(results)<=2 or len(results)%10==0:
            print(f'{args.max_side}: {len(results)}/71; correct={sum(r["correct"] for r in results)}; grid={grid}',flush=True)
        del inputs,output,ids,user,images
    assert all(digest(Path(p))==h for p,h in hashes.items())
    save_json(run/'status.json',{'status':'complete','completed':71,'total':71,'pid':os.getpid(),
        'elapsed_seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--max-side',type=int,choices=[512,1024],required=True)
    parser.add_argument('--output',type=Path,required=True)
    evaluate(parser.parse_args())
