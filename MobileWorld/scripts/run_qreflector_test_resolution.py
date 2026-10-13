#!/usr/bin/env python3
"""Run the authorized 1024-pixel test and compare fixed predictions automatically."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import qreflector_geometry_prompt as geometry
from freeze_qreflector_review_labels import metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
OLD = ROOT / 'artifacts/training/qreflector_sol_round2_geometry_20261009_gpu2/test'
ADAPTER = OLD.parent / 'train/final'
PROCESSOR = ROOT / 'artifacts/training/qreflector_sol_round1_v9_epoch9_20261008_gpu7/train/final'
REVISION = ROOT / 'artifacts/analysis/qreflector_label_revision_r1_20261009'
RUN = ROOT / 'artifacts/evaluations/qreflector_round2_test_resolution_20261009_gpu5'
GPU = 'GPU-f25fb1a7-2bc8-d28f-8264-315da6a4f5b5'
PYTHON = ROOT / 'artifacts/envs/qreflector-native/bin/python'
EVALUATOR = ROOT / 'scripts/eval_qreflector_test_resolution.py'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify(plan):
    for path, expected in plan['protected_files'].items():
        assert sha(path) == expected, f'Protected file changed: {path}'


def preflight():
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'side_1024').mkdir()
    source = rows(DATA / 'test.jsonl')
    labels = rows(DATA / 'test.labels.jsonl')
    old = rows(OLD / 'predictions.jsonl')
    assert len(source) == len(labels) == len(old) == 1082
    old_manifest = read(OLD / 'manifest.json')
    assert old_manifest['adapter'] == str(ADAPTER)
    assert old_manifest['adapter_sha256'] == sha(ADAPTER / 'adapter_model.safetensors')
    assert old_manifest['data_sha256'] == sha(DATA / 'test.jsonl')
    assert old_manifest['prompt_template_sha256'] == sha(DATA / 'prompt_template.txt')
    assert old_manifest['geometry_builder_sha256'] == sha(Path(geometry.__file__))
    assert old_manifest['image_max_side'] == 512 and old_manifest['max_new_tokens'] == 1024
    assert not old_manifest['do_sample'] and old_manifest['batch_size'] == 1
    assert old_manifest['dtype'] == 'bfloat16' and old_manifest['quantization'] is None
    revision = read(REVISION / 'evaluation_manifest.json')
    assert revision['labels_sha256'] == sha(REVISION / 'test.labels.jsonl')
    assert revision['original_labels_sha256'] == sha(DATA / 'test.labels.jsonl')
    files = [DATA / name for name in ('test.jsonl', 'test.labels.jsonl', 'test.inputs.jsonl', 'val.jsonl', 'train.jsonl', 'prompt_template.txt')]
    files += [OLD / name for name in ('manifest.json', 'predictions.jsonl', 'summary.json')]
    files += [ADAPTER / 'adapter_model.safetensors', ADAPTER / 'adapter_config.json',
              REVISION / 'test.labels.jsonl', REVISION / 'evaluation_manifest.json',
              EVALUATOR, Path(__file__), Path(geometry.__file__),
              ROOT / 'scripts/qreflector_opsd.py', ROOT / 'scripts/qreflector_sft_bf16.py',
              ROOT / 'scripts/eval_gui_reflection.py']
    processor_files = []
    for name in ('preprocessor_config.json', 'chat_template.jinja', 'tokenizer.json',
                 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'added_tokens.json',
                 'special_tokens_map.json', 'video_preprocessor_config.json'):
        a, b = PROCESSOR / name, ADAPTER / name
        if a.exists() or b.exists():
            assert a.exists() and b.exists() and sha(a) == sha(b), name
            processor_files.append(name)
            files += [a, b]
    protected = {str(p): sha(p) for p in files}
    for sid, (row, label, prediction) in enumerate(zip(source, labels, old), 1):
        assert prediction['sample_id'] == sid and prediction['source_sample_id'] == label['sample_id']
        assert prediction['expected'] == label['answer'].lower()
        user = row['messages'][0]
        assert user['role'] == 'user' and [c['type'] for c in user['content']] == ['text', 'image', 'image']
        prompt = geometry.augment_row(row, DATA, 512)['messages'][0]['content'][0]['text']
        assert hashlib.sha256(prompt.encode()).hexdigest() == prediction['prompt_sha256'], sid
        for item in user['content']:
            if item['type'] == 'image':
                path = (DATA / item['image']).resolve()
                protected[str(path)] = sha(path)
    plan = {
        'created_unix': time.time(), 'gpu_uuid': GPU, 'gpu_index': 5,
        'checkpoint': str(ADAPTER), 'samples': 1082,
        'fresh_inference': '1024 long side only; original 512 predictions reused',
        'primary_labels': str(DATA / 'test.labels.jsonl'),
        'supplemental_labels': str(REVISION / 'test.labels.jsonl'),
        'primary_labels_policy': 'Original benchmark unchanged; separate assistant-reviewed labels are sensitivity analysis only.',
        'primary_scoring': 'Frozen parse_verdict, including invalid responses as errors.',
        'answer_only_supplement': 'For both resolutions, retain valid parsed verdicts; for invalid JSON only, recover one unambiguous terminal answer field. No other text is repaired.',
        'original_512_prompt_hashes_verified': 1082,
        'processor_files_identical': processor_files,
        'memory_cap_gib': 32, 'reserve_gib': 8,
        'protected_files': protected,
        'no_training': True, 'no_external_model_api': True,
        'no_process_signals_or_device_settings': True,
    }
    write(RUN / 'preflight.json', plan)
    write(RUN / 'pipeline_status.json', {'status': 'prepared', 'updated_unix': time.time()})
    print(json.dumps({'prepared': str(RUN), 'samples': 1082, 'protected_files': len(protected)}))


def answers(records, tolerant=False):
    output = {}
    for r in records:
        verdict = r['predicted']
        if tolerant and verdict is None:
            all_fields = re.findall(r'"answer"\s*:\s*"(Yes|No)"', r['full_response'])
            terminal = re.findall(r'"answer"\s*:\s*"(Yes|No)"\s*}\s*$', r['full_response'])
            if len(all_fields) == 1 and all_fields == terminal:
                verdict = terminal[0].lower()
        output[r['sample_id']] = verdict
    return output


def report():
    plan = read(RUN / 'preflight.json')
    verify(plan)
    assert read(RUN / 'side_1024/status.json')['status'] == 'complete'
    old = rows(OLD / 'predictions.jsonl')
    new = rows(RUN / 'side_1024/predictions.jsonl')
    source = rows(DATA / 'test.jsonl')
    assert len(old) == len(new) == len(source) == 1082
    for i, (a, b, row) in enumerate(zip(old, new, source), 1):
        assert a['sample_id'] == b['sample_id'] == i
        assert a['source_sample_id'] == b['source_sample_id']
        assert a['expected'] == b['expected']
        prompt = geometry.augment_row(row, DATA, 1024)['messages'][0]['content'][0]['text']
        assert hashlib.sha256(prompt.encode()).hexdigest() == b['prompt_sha256']
        assert b['image_geometry'] == geometry.geometry(row, DATA, 1024)
    assert sum(r['correct'] for r in old) == read(OLD / 'summary.json')['correct'] == 863
    label_sets = {
        'original': {i: r['answer'].lower() for i, r in enumerate(rows(DATA / 'test.labels.jsonl'), 1)},
        'review_r1': {i: r['answer'].lower() for i, r in enumerate(rows(REVISION / 'test.labels.jsonl'), 1)},
    }
    result_rows, pairs = [], []
    for version, labels in label_sets.items():
        for policy, tolerant in [('original_parser', False), ('answer_only', True)]:
            a, b = answers(old, tolerant), answers(new, tolerant)
            fixed = [i for i in labels if a[i] != labels[i] and b[i] == labels[i]]
            regressed = [i for i in labels if a[i] == labels[i] and b[i] != labels[i]]
            for side, pred in [(512, a), (1024, b)]:
                result_rows.append({'labels': version, 'scoring': policy, 'long_side': side, **metrics(pred, labels)})
            pairs.append({'labels': version, 'scoring': policy,
                          'fixed_ids': fixed, 'regressed_ids': regressed,
                          'fixed_count': len(fixed), 'regressed_count': len(regressed),
                          'net_correct_change': len(fixed) - len(regressed),
                          'delta_accuracy_pp': 100 * (len(fixed) - len(regressed)) / 1082,
                          'both_wrong_ids': [i for i in labels if a[i] != labels[i] and b[i] != labels[i]]})
    format_rows = []
    for side, rs in [(512, old), (1024, new)]:
        recovered = [r['sample_id'] for r in rs if r['predicted'] is None and answers([r], True)[r['sample_id']] is not None]
        format_rows.append({'long_side': side, 'invalid_answers': sum(r['predicted'] is None for r in rs),
            'invalid_four_field_json': sum(not r['valid_four_field_json'] for r in rs),
            'truncated_outputs': sum(r['at_generation_limit'] for r in rs),
            'answer_field_recovered_ids': recovered,
            'mean_input_tokens': statistics.mean(r['input_tokens'] for r in rs),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rs),
            'mean_reported_latency_seconds': statistics.mean(r['latency_seconds'] for r in rs)})
    changed = [{'sample_id': a['sample_id'], 'original_expected': a['expected'],
                'revised_expected': label_sets['review_r1'][a['sample_id']],
                'side_512': a, 'side_1024': b}
               for a, b in zip(old, new) if a['predicted'] != b['predicted'] or
               answers([a], True)[a['sample_id']] != answers([b], True)[b['sample_id']]]
    report_data = {'checkpoint': str(ADAPTER), 'samples': 1082, 'metrics': result_rows,
        'paired_comparisons': pairs, 'output_diagnostics': format_rows,
        'verification': {'all_protected_sources_unchanged': True, 'all_prompt_hashes_verified': True,
            'original_512_prediction_sha256': sha(OLD / 'predictions.jsonl'),
            'new_1024_prediction_sha256': sha(RUN / 'side_1024/predictions.jsonl')},
        'limitations': ['Resolution was selected after a 71-case validation pilot.',
            'Pixel resolution and corresponding geometry dimension text change together.',
            'Reviewed labels are a post-hoc same-assistant sensitivity version, not independently adjudicated ground truth.',
            '512 and 1024 ran at different times/devices; mean reported latency is descriptive and excludes different amounts of preprocessing.',
            'A correct Yes/No answer does not establish explanation or condition quality.']}
    write(RUN / 'comparison.json', report_data)
    write(RUN / 'changed_cases.json', changed)
    with (RUN / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    lines = ['Fixed Sol round-2 checkpoint: 512 vs 1024 on 1082 test cases', '',
             'labels | scoring | long side | correct | accuracy | success recognition | failure recognition | invalid']
    lines += [f'{r["labels"]} | {r["scoring"]} | {r["long_side"]} | {r["correct"]}/1082 | {100*r["accuracy"]:.2f}% | {100*r["success_recognition_rate"]:.2f}% | {100*r["failure_recognition_rate"]:.2f}% | {r["invalid"]}' for r in result_rows]
    lines += ['', 'Paired comparisons:']
    lines += [f'{p["labels"]}/{p["scoring"]}: fixed {p["fixed_count"]}, new errors {p["regressed_count"]}, net {p["net_correct_change"]:+d} ({p["delta_accuracy_pp"]:+.2f} pp)' for p in pairs]
    lines += ['', 'Original test labels, images, weights and 512 outputs remain unchanged.', *report_data['limitations']]
    (RUN / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def pipeline():
    plan = read(RUN / 'preflight.json')
    with (RUN / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(plan)
        if read(RUN / 'pipeline_status.json')['status'] == 'complete':
            return
        command = [str(PYTHON), '-u', str(EVALUATOR), '--gpu-uuid', GPU,
                   '--max-side', '1024', '--output', str(RUN / 'side_1024')]
        try:
            with (RUN / 'side_1024/worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                write(RUN / 'pipeline_status.json', {'status': 'running', 'stage': 'test',
                    'pid': os.getpid(), 'worker_pid': child.pid, 'updated_unix': time.time(), 'command': command})
                code = child.wait()
            if code:
                raise RuntimeError(f'Evaluation exited with code {code}; inspect side_1024/worker.log')
            write(RUN / 'pipeline_status.json', {'status': 'running', 'stage': 'report',
                'pid': os.getpid(), 'worker_pid': child.pid, 'updated_unix': time.time()})
            report()
            gpu_rows = subprocess.check_output(['nvidia-smi', '-i', GPU,
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
            assert str(child.pid) not in [r.strip() for r in gpu_rows]
            write(RUN / 'gpu_release_verification.json', {'checked_at_unix': time.time(),
                'worker_pid': child.pid, 'worker_exited_naturally': True,
                'worker_absent_from_gpu_processes': True, 'processes_signaled': []})
            write(RUN / 'pipeline_status.json', {'status': 'complete', 'stage': 'complete',
                'pid': os.getpid(), 'updated_unix': time.time(), 'report': str(RUN / 'REPORT.txt')})
        except BaseException as exc:
            write(RUN / 'pipeline_status.json', {'status': 'failed', 'pid': os.getpid(),
                'updated_unix': time.time(), 'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'pipeline', 'report'])
    args = parser.parse_args()
    {'preflight': preflight, 'pipeline': pipeline, 'report': report}[args.stage]()
