#!/usr/bin/env python3
"""Audit complete runs against the pinned native benchmark and compare metrics."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from eval_gui_reflection import ROOT, parse_verdict, save_json, summarize
from eval_gui_reflection_native import official_template, sha256_file


def main():
    data_dir = ROOT / 'artifacts/datasets/gui_reflection_task_suite'
    output_root = ROOT / 'artifacts/evaluations/gui_reflection'
    data = json.loads((data_dir / 'dataset_manifest.json').read_text())
    annotation = data_dir / 'action_verification.jsonl'
    assert sha256_file(annotation) == data['annotation_sha256']
    pairs = [json.loads(line) for line in annotation.read_text().splitlines()]
    assert len(pairs) == 603
    template = official_template(ROOT / 'artifacts/vendor/GUI_Reflection/internvl_chat/eval/eval_tasks/reflection_action_verification_infer.py')
    expected = []
    for index, pair in enumerate(pairs, 1):
        for kind, label in [('positive', 'yes'), ('negative', 'no')]:
            expected.append({'pair_id': index, 'kind': kind, 'expected': label,
                             'purpose': pair[kind + '_action_purpose'], 'images': pair['images']})
    image_manifest = {item['name']: item for item in data['images']}
    assert set(image_manifest) == {name for pair in pairs for name in pair['images']}
    for name, item in image_manifest.items():
        assert sha256_file(data_dir / 'images' / name) == item['sha256'], name

    results = {}
    reports = {}
    for name, directory in [('gui_reflection', 'native_gui_reflection_gpu0'),
                            ('qreflector', 'native_qreflector_dpo_grpo_gpu0')]:
        path = output_root / directory
        manifest = json.loads((path / 'manifest.json').read_text())
        status = json.loads((path / 'status.json').read_text())
        assert status['status'] == 'complete' and status['completed'] == 1206
        assert manifest['annotation_sha256'] == data['annotation_sha256']
        assert manifest['dataset_manifest_sha256'] == sha256_file(data_dir / 'dataset_manifest.json')
        assert manifest['evaluator_sha256'] == sha256_file(path / 'evaluator_source.py')
        assert manifest['prompt_template'] == template
        rows = [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(expected) == 1206
        for index, (row, sample) in enumerate(zip(rows, expected), 1):
            assert row['sample_id'] == index
            assert all(row[key] == value for key, value in sample.items())
            prompt = template.format(action_purpose=sample['purpose'])
            assert row['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            assert row['predicted'] == parse_verdict(row['full_response'])
            assert row['correct'] == (row['predicted'] == sample['expected'])
            assert row['official_correct'] == row['full_response'].strip().lower().startswith(sample['expected'])
        report = summarize(rows)
        saved_summary = json.loads((path / 'summary.json').read_text())
        assert all(saved_summary[key] == value for key, value in report.items())
        report.update({
            'official_prefix_accuracy': sum(row['official_correct'] for row in rows) / len(rows),
            'invalid_sample_ids': [row['sample_id'] for row in rows if row['predicted'] is None],
            'accuracy_if_all_invalid_outputs_were_correct': (report['correct'] + report['invalid_outputs']) / len(rows),
            'both_purposes_correct_pairs': sum(rows[i]['correct'] and rows[i + 1]['correct'] for i in range(0, len(rows), 2)),
            'mean_latency_seconds': statistics.mean(row['latency_seconds'] for row in rows),
            'median_latency_seconds': statistics.median(row['latency_seconds'] for row in rows),
            'input_token_range': [min(row['input_tokens'] for row in rows), max(row['input_tokens'] for row in rows)],
            'output_token_counts': dict(Counter(row['output_tokens'] for row in rows)),
            'max_token_limit_hits': sum(row['output_tokens'] >= manifest['generation']['max_new_tokens'] for row in rows),
            'peak_allocated_gib': status['peak_allocated_gib'],
            'elapsed_seconds': status['elapsed_seconds'],
            'predictions_sha256': sha256_file(path / 'predictions.jsonl'),
            'manifest': str(path / 'manifest.json'),
        })
        if name == 'qreflector':
            audit = json.loads((path / 'adapter_validation.json').read_text())
            assert audit['all_saved_tensors_match_loaded'] and audit['tensor_count'] == 504
        results[name] = rows
        reports[name] = report
        save_json(path / 'invalid_outputs.json', [row for row in rows if row['predicted'] is None])
        save_json(path / 'result_validation.json', {'passed': True, 'checked_judgments': 1206,
                  'verified_image_hashes': len(image_manifest), 'official_prefix_equals_strict_accuracy':
                  report['official_prefix_accuracy'] == report['accuracy']})
    outcomes = Counter((a['correct'], b['correct']) for a, b in zip(results['gui_reflection'], results['qreflector']))
    comparison = {
        'benchmark': data['benchmark_repo'], 'revision': data['benchmark_revision'],
        'pairs': 603, 'judgments': 1206, 'positive': 603, 'negative': 603,
        'label_semantics': 'Fulfilled versus unfulfilled action purpose on the same screenshot pairs; not overall task success.',
        'models': reports,
        'paired_outcomes': {'both_correct': outcomes[(True, True)], 'gui_only_correct': outcomes[(True, False)],
                            'qreflector_only_correct': outcomes[(False, True)], 'both_wrong': outcomes[(False, False)]},
        'paper_gui_reflection_reference': {'accuracy': 0.8756, 'positive_recall': 0.8159,
                                           'negative_recall': 0.9353, 'source': 'https://arxiv.org/html/2506.08012v1#S3.T1'},
        'protocol': 'Frozen models; same official purpose prompt and original screenshots; native model-specific image preprocessing; greedy decoding.',
        'caveats': ['Q-Reflector uses the MobileWorld DPO+GRPO checkpoint, not the separate GRPO checkpoint associated with 82.16%.',
                    'Q-Reflector is evaluated through Transformers+PEFT with its configured 4-bit base; Unsloth runtime patches are not used.',
                    'Negative purposes are synthetic mismatches on observed transitions; these metrics are not directly interchangeable with AndroidWorld or closed-loop success rates.'],
    }
    save_json(output_root / 'native_comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
