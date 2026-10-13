#!/usr/bin/env python3
"""Sequential, isolated MAI and GUI-Owl BF16 evaluations on the frozen 1082 test."""
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/evaluations/mai_owl_v9_1082_20261011_gpu6'
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
REFERENCE = ROOT / 'artifacts/evaluations/qreflector_round2_test_resolution_20261009_gpu5/side_1024'
GPU = 'GPU-5182a8a6-4d6f-af6f-0415-55fd854c2989'
PYTHON = ROOT / 'artifacts/envs/qreflector-native/bin/python'
sys.path.insert(0, str(RUN / 'source' if (RUN / 'source').is_dir() else ROOT / 'scripts'))
from eval_gui_reflection import parse_verdict, summarize, save_json, gpu_snapshot
from qreflector_sft_bf16 import digest, read_rows, load_images, strict_structure, check_bf16_base
import qreflector_geometry_prompt as geometry


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), uid=p.stat().st_uid, start_ticks=fields[19], state=fields[0])


def gpu_processes():
    raw = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in raw.splitlines():
        pid, memory = line.split(',')
        try:
            rows.append(dict(**identity(int(pid)), memory_mib=int(memory)))
        except FileNotFoundError:
            pass
    return rows


def gpu_free_gib():
    return int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip()) / 1024


def verify(plan):
    for filename, expected in plan['pinned_files'].items():
        assert digest(Path(filename)) == expected, f'Pinned file changed: {filename}'
    for filename, expected in plan['image_sha256'].items():
        assert digest(Path(filename)) == expected, f'Test image changed: {filename}'
    for filename, expected in plan['weight_stats'].items():
        stat = Path(filename).stat()
        assert [stat.st_size, stat.st_mtime_ns] == expected, f'Model shard changed: {filename}'


def recover_answer(raw, prediction):
    if prediction is not None:
        return prediction
    fields = re.findall(r'"answer"\s*:\s*"(Yes|No)"', raw)
    terminal = re.findall(r'"answer"\s*:\s*"(Yes|No)"\s*}\s*$', raw)
    return terminal[0].lower() if len(fields) == 1 and fields == terminal else None


def report(out, results):
    summary = summarize(results)
    summary['tokens'] = dict(input=sum(r['input_tokens'] for r in results),
                            output=sum(r['output_tokens'] for r in results))
    summary['at_generation_limit'] = sum(r['at_generation_limit'] for r in results)
    summary['invalid_four_field_json'] = sum(not r['valid_four_field_json'] for r in results)
    save_json(out / 'summary.json', summary)
    alternate = [{**r, 'predicted': r['answer_only_predicted'],
                  'correct': r['answer_only_predicted'] == r['expected']} for r in results]
    save_json(out / 'summary_answer_only_supplement.json', summarize(alternate))


def prepare():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'source').mkdir()
    (RUN / 'tmp').mkdir()
    for name in ['eval_gui_reflection.py', 'qreflector_sft_bf16.py', 'qreflector_geometry_prompt.py']:
        shutil.copy2(ROOT / 'scripts' / name, RUN / 'source' / name)
    shutil.copy2(Path(__file__).resolve(), RUN / 'experiment.py')
    model_runs = {
        'mai': ROOT / 'artifacts/evaluations/mai_self_reflection/mobileworld_mai_self117_v9_geometry_noearly_20261011_gpu2_p4',
        'owl': ROOT / 'artifacts/evaluations/gui_owl_self_reflection/mobileworld_gui_owl_self117_shared_noearly_20261010_gpu7_p4',
    }
    models = {}
    pinned = [DATA / n for n in ['test.inputs.jsonl', 'test.labels.jsonl', 'test.jsonl',
                                'DATASET_MANIFEST.json', 'prompt_template.txt']]
    pinned += list((RUN / 'source').glob('*.py')) + [RUN / 'experiment.py']
    pinned += [REFERENCE / 'manifest.json', REFERENCE / 'predictions.jsonl']
    weight_stats = {}
    for name, parent in model_runs.items():
        cfg = json.loads((parent / 'config.json').read_text())
        model = Path(cfg['model_path'])
        models[name] = dict(repo=cfg['actor_model'], revision=cfg['actor_revision'], path=str(model),
                            source_run=str(parent))
        metadata = [p for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')]
        pinned += metadata
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        for filename in set(index['weight_map'].values()):
            shard = model / filename
            stat = shard.stat()
            weight_stats[str(shard)] = [stat.st_size, stat.st_mtime_ns]
        (RUN / name).mkdir()
    rows = read_rows(DATA / 'test.inputs.jsonl')
    labels = read_rows(DATA / 'test.labels.jsonl')
    reference = read_rows(REFERENCE / 'predictions.jsonl')
    full = read_rows(DATA / 'test.jsonl')
    dataset = json.loads((DATA / 'DATASET_MANIFEST.json').read_text())
    for filename in ['test.inputs.jsonl', 'test.labels.jsonl', 'test.jsonl', 'prompt_template.txt']:
        assert digest(DATA / filename) == dataset['artifact_sha256'][filename]
    assert len(rows) == len(labels) == len(reference) == len(full) == 1082
    image_hashes, inputs = {}, []
    for i, (row, label, prior, labeled) in enumerate(zip(rows, labels, reference, full), 1):
        assert [m['role'] for m in row['messages']] == ['user']
        assert row['messages'][0] == labeled['messages'][0]
        assert label['answer'].lower() == prior['expected']
        assert label['sample_id'] == prior['source_sample_id'] and prior['sample_id'] == i
        assert set(geometry.context_of(row)) == {'goal', 'action', 'history'}
        augmented = geometry.augment_row(row, DATA, 1024)
        text = augmented['messages'][0]['content'][0]['text']
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()
        assert prompt_hash == prior['prompt_sha256'], i
        info = geometry.geometry(row, DATA, 1024)
        assert info == prior['image_geometry']
        for part in row['messages'][0]['content']:
            if part['type'] == 'image':
                path = (DATA / part['image']).resolve()
                if str(path) not in image_hashes:
                    image_hashes[str(path)] = digest(path)
        inputs.append(dict(sample_id=i, source_sample_id=label['sample_id'],
                           prompt_sha256=prompt_hash, image_geometry=info))
    (RUN / 'inputs_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in inputs))
    pinned.append(RUN / 'inputs_manifest.jsonl')
    plan = dict(gpu_index=6, gpu_uuid=GPU, models=models, order=['mai', 'owl'], samples=1082,
                dtype='bfloat16', quantization=None, batch_size=1, max_new_tokens=1024,
                max_sequence_length=8192, image_max_side=1024, do_sample=False, seed=42,
                memory_cap_gib=32, minimum_free_before_load_gib=44, reserve_gib=12,
                data=str(DATA), labels=str(DATA / 'test.labels.jsonl'),
                label_policy='Unchanged original test labels; no relabeling; no reference answers or explanations in model inputs.',
                input_policy='Exact per-row v9 + dynamic-geometry prompt hashes and images match the existing Q-Reflector round2 1024 test.',
                processor_policy='Each model uses its own published processor/chat template; sampling disabled explicitly.',
                primary_scoring='Existing frozen parse_verdict; invalid responses count as wrong. Score final Yes/No only.',
                supplemental_scoring='Separately report recovery of one unambiguous terminal answer field, matching the prior resolution report.',
                process_policy='Sequential private inference children; no signals to any process, no services/containers/device setting changes.',
                pinned_files={str(p.resolve()): digest(p) for p in pinned}, image_sha256=image_hashes,
                weight_stats=weight_stats, reference_run=str(REFERENCE), created_at_unix=time.time())
    save_json(RUN / 'plan.json', plan)
    save_json(RUN / 'pipeline_status.json', dict(status='prepared', order=['mai', 'owl'], total_each=1082))
    print(json.dumps(dict(prepared=str(RUN), exact_prompt_matches=1082, unique_images=len(image_hashes))))


def cpu_validate():
    os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1')
    plan = json.loads((RUN / 'plan.json').read_text())
    verify(plan)
    from transformers import AutoProcessor
    rows = read_rows(DATA / 'test.inputs.jsonl')
    first = geometry.augment_row(rows[0], DATA, 1024)['messages'][0]
    for part, image in zip([p for p in first['content'] if p['type'] == 'image'], load_images(rows[0], DATA, 1024)):
        part['image'] = image
    records = {}
    for name, model in plan['models'].items():
        processor = AutoProcessor.from_pretrained(model['path'], local_files_only=True)
        inputs = processor.apply_chat_template([first], add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors='pt', truncation=False)
        n = inputs['input_ids'].shape[-1]
        assert n + 1024 <= 8192 and len(inputs['image_grid_thw']) == 2
        records[name] = dict(processor=type(processor).__name__, input_tokens=n,
                             image_grid_thw=inputs['image_grid_thw'].tolist())
    assert parse_verdict('{"answer":"Yes"}') == 'yes'
    assert parse_verdict('{"reason":"Yes is unsupported","answer":"No"}') == 'no'
    assert parse_verdict('{"precondition":"a","postcondition":"b","reason":"c"}') is None
    assert recover_answer('"answer": "Yes"}', None) == 'yes'
    save_json(RUN / 'cpu_validation.json', dict(passed=True, no_gpu=True, no_network=True, models=records,
                                               original_labels_and_all_1082_prompt_hashes_verified=True))
    print(json.dumps(records))


def worker(name):
    plan = json.loads((RUN / 'plan.json').read_text())
    out = RUN / name
    assert not (out / 'predictions.jsonl').exists(), 'No implicit overwrite/resume'
    verify(plan)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      HF_HOME=str(out / 'cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', PYTORCH_ALLOC_CONF='expandable_segments:True')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == GPU.removeprefix('GPU-')
    free, total = torch.cuda.mem_get_info(0)
    assert free >= plan['minimum_free_before_load_gib'] * 2**30, 'Insufficient headroom; other processes untouched'
    torch.cuda.set_per_process_memory_fraction(plan['memory_cap_gib'] * 2**30 / total, 0)
    torch.manual_seed(42)
    rows, labels = read_rows(DATA / 'test.inputs.jsonl'), read_rows(DATA / 'test.labels.jsonl')
    inputs_manifest = read_rows(RUN / 'inputs_manifest.jsonl')
    model_info = plan['models'][name]
    manifest = dict(**{k:v for k,v in plan.items() if k not in ['pinned_files', 'image_sha256', 'weight_stats', 'models']},
                    model=model_info, pid=os.getpid(), plan_sha256=digest(RUN / 'plan.json'),
                    versions={p:importlib.metadata.version(p) for p in ['torch', 'transformers', 'accelerate']})
    save_json(out / 'manifest.json', manifest)
    save_json(out / 'status.json', dict(status='loading', completed=0, total=1082, pid=os.getpid()))
    save_json(out / 'gpu_before.json', gpu_snapshot(GPU))
    processor = AutoProcessor.from_pretrained(model_info['path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_info['path'], dtype=torch.bfloat16,
                device_map={'':'cuda:0'}, local_files_only=True, attn_implementation='sdpa')
    model.requires_grad_(False)
    model.eval()
    save_json(out / 'model_validation.json', check_bf16_base(model))
    results, started = [], time.monotonic()
    for i, row in enumerate(rows):
        while torch.cuda.mem_get_info()[0] < plan['reserve_gib'] * 2**30:
            torch.cuda.empty_cache()
            save_json(out / 'status.json', dict(status='waiting_for_headroom', completed=len(results),
                total=1082, pid=os.getpid(), updated=time.time()))
            time.sleep(10)
        augmented = geometry.augment_row(row, DATA, 1024)
        user = copy.deepcopy(augmented['messages'][0])
        text = user['content'][0]['text']
        assert hashlib.sha256(text.encode()).hexdigest() == inputs_manifest[i]['prompt_sha256']
        images = load_images(row, DATA, 1024)
        info = inputs_manifest[i]['image_geometry']
        assert [list(im.size) for im in images] == [[g['input_width'],g['input_height']] for g in info['images']]
        for part, image in zip([p for p in user['content'] if p['type']=='image'], images):
            part['image'] = image
        begun = time.monotonic()
        inputs = processor.apply_chat_template([user], add_generation_prompt=True, tokenize=True,
                      return_dict=True, return_tensors='pt', truncation=False)
        grid = inputs['image_grid_thw'].tolist()
        n = inputs['input_ids'].shape[-1]
        assert n + 1024 <= 8192
        inputs = {k:v.to('cuda:0') for k,v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=1,
                                    temperature=None, top_p=None, top_k=None, repetition_penalty=1.)
        ids = output[0,n:]
        response = processor.decode(ids, skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        predicted = parse_verdict(response)
        expected = labels[i]['answer'].lower()
        rec = dict(sample_id=i+1, source_sample_id=labels[i]['sample_id'], expected=expected,
                   predicted=predicted, correct=predicted==expected, full_response=response,
                   answer_only_predicted=recover_answer(response, predicted),
                   valid_four_field_json=strict_structure(response),
                   prompt_sha256=inputs_manifest[i]['prompt_sha256'], image_geometry=info,
                   image_grid_thw=grid, input_tokens=int(n), output_tokens=len(ids),
                   at_generation_limit=len(ids)>=1024, latency_seconds=time.monotonic()-begun)
        with (out / 'predictions.jsonl').open('a') as handle:
            handle.write(json.dumps(rec, ensure_ascii=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        results.append(rec)
        report(out, results)
        save_json(out / 'status.json', dict(status='running', completed=len(results), total=1082,
            pid=os.getpid(), elapsed_seconds=time.monotonic()-started, updated=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        if len(results)<=3 or len(results)%25==0:
            print(f'{name}: {len(results)}/1082, correct={sum(r["correct"] for r in results)}, output_tokens={len(ids)}', flush=True)
        del inputs, output, ids, images, user
    verify(plan)
    save_json(out / 'status.json', dict(status='complete', completed=1082, total=1082, pid=os.getpid(),
        updated=time.time(), elapsed_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def pipeline():
    with (RUN / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((RUN / 'plan.json').read_text())
        verify(plan)
        assert json.loads((RUN / 'cpu_validation.json').read_text())['passed']
        protected = gpu_processes()
        save_json(RUN / 'protected_processes.json', protected)
        save_json(RUN / 'gpu_before.json', gpu_snapshot(GPU))
        for name in plan['order']:
            while gpu_free_gib() < plan['minimum_free_before_load_gib']:
                save_json(RUN / 'pipeline_status.json', dict(status='waiting_for_headroom', next_model=name,
                                                          free_gib=gpu_free_gib(), updated=time.time()))
                time.sleep(10)
            save_json(RUN / 'pipeline_status.json', dict(status='running', current_model=name,
                remaining=plan['order'][plan['order'].index(name):], updated=time.time()))
            with (RUN / name / 'run.log').open('x') as log:
                proc = subprocess.Popen([str(PYTHON), str(RUN / 'experiment.py'), 'worker', '--model', name],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                save_json(RUN / name / 'process.json', identity(proc.pid))
                rc = proc.wait()
            status = json.loads((RUN / name / 'status.json').read_text())
            assert rc == 0 and status['status']=='complete' and status['completed']==1082, f'{name} failed, see run.log'
            # The child exits naturally; no signal is sent, even for release.
            deadline = time.monotonic()+60
            while proc.pid in {p['pid'] for p in gpu_processes()}:
                assert time.monotonic()<deadline, 'Own worker memory not yet released'
                time.sleep(2)
            save_json(RUN / name / 'gpu_after_release.json', gpu_snapshot(GPU))
            checks=[]
            for old in protected:
                try:
                    cur=identity(old['pid'])
                    checks.append(dict(pid=old['pid'], same_identity=cur['uid']==old['uid'] and cur['start_ticks']==old['start_ticks']))
                except FileNotFoundError:
                    checks.append(dict(pid=old['pid'], exited_independently=True))
            save_json(RUN / name / 'protected_process_checks.json', checks)
        save_json(RUN / 'comparison.json', {n:json.loads((RUN/n/'summary.json').read_text()) for n in plan['order']})
        save_json(RUN / 'pipeline_status.json', dict(status='complete', completed_models=plan['order'], updated=time.time()))


def launch():
    assert not (RUN / 'launch.json').exists(), 'Refuse duplicate launch'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(RUN / 'tmp'), USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (RUN / 'pipeline.log').open('x') as log:
        proc = subprocess.Popen(['nice', '-n', '10', str(PYTHON), str(RUN / 'experiment.py'), 'pipeline'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(**identity(proc.pid), run=str(RUN))
    save_json(RUN / 'launch.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','cpu_validate','launch','pipeline','worker'])
    parser.add_argument('--model', choices=['mai','owl'])
    args=parser.parse_args()
    try:
        if args.command=='worker': worker(args.model)
        else: globals()[args.command]()
    except BaseException as exc:
        if args.command in ['worker','pipeline']:
            location=RUN/args.model/'status.json' if args.command=='worker' else RUN/'pipeline_status.json'
            old=json.loads(location.read_text()) if location.exists() else {}
            save_json(location,{**old,'status':'failed','error':f'{type(exc).__name__}: {exc}','updated':time.time()})
        raise
