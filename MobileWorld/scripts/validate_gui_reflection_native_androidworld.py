#!/usr/bin/env python3
"""Audit a completed native-prompt AndroidWorld run and compare its v6 baseline."""

import argparse
import json
from collections import Counter
from pathlib import Path

from eval_gui_reflection import ROOT, parse_verdict, save_json, summarize
from eval_gui_reflection_native import sha256_file
from eval_gui_reflection_native_androidworld import DATA, native_input, official_template, PROMPT_SOURCE, read_rows, verify_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert ROOT / 'artifacts/evaluations/gui_reflection' in output.parents
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'complete' and status['completed'] == 1082
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['evaluator_sha256'] == sha256_file(output / 'evaluator_source.py')
    for name, expected in manifest['helpers_sha256'].items():
        assert sha256_file(ROOT / 'scripts' / name) == expected
    assert sha256_file(DATA / 'test.inputs.jsonl') == manifest['input_sha256']
    assert sha256_file(DATA / 'test.labels.jsonl') == manifest['labels_sha256']
    assert sha256_file(PROMPT_SOURCE) == manifest['prompt_source_sha256']
    template = official_template(PROMPT_SOURCE)
    prepared = [dict(sample_id=i, **native_input(row, template))
                for i, row in enumerate(read_rows(DATA / 'test.inputs.jsonl'), 1)]
    assert prepared == read_rows(output / 'native_inputs.jsonl')
    predictions = read_rows(output / 'predictions.jsonl')
    labels = read_rows(DATA / 'test.labels.jsonl')
    assert len(predictions) == len(prepared) == len(labels) == 1082
    verify_saved(predictions, prepared, labels)
    for prediction, expected in zip(predictions, prepared, strict=True):
        assert all(prediction[key] == expected[key] for key in ['purpose', 'images', 'prompt_sha256'])
        assert prediction['official_correct'] == prediction['full_response'].strip().lower().startswith(prediction['expected'])
    native_metrics = summarize(predictions)
    saved_summary = json.loads((output / 'summary.json').read_text())
    assert all(saved_summary[key] == value for key, value in native_metrics.items())

    old = ROOT / 'artifacts/training/qreflector_sft_v6_20261002_gpu0/evaluations/gui-reflection-8b'
    old_manifest = json.loads((old / 'manifest.json').read_text())
    for field in ['model_repo', 'model_revision', 'official_code_revision', 'image_policy', 'dtype', 'batch_size']:
        assert manifest[field] == old_manifest[field], field
    for field in ['max_new_tokens', 'do_sample', 'num_beams']:
        assert manifest['generation'][field] == old_manifest['generation'][field]
    baseline = read_rows(old / 'predictions.jsonl')
    assert len(baseline) == 1082
    for a, b in zip(predictions, baseline, strict=True):
        assert a['sample_id'] == b['sample_id'] and a['expected'] == b['expected']
        assert [(DATA / path).resolve() for path in a['images']] == [
            (Path(old_manifest['data']).parent / path).resolve() for path in b['images']]
        assert b['predicted'] == parse_verdict(b['full_response'])
        assert b['correct'] == (b['predicted'] == b['expected'])
    baseline_metrics = summarize(baseline)
    pairs = Counter((a['correct'], b['correct']) for a, b in zip(predictions, baseline))
    report = {
        'audit_passed': True, 'samples': 1082, 'native_prompt': native_metrics,
        'v6_shared_prompt': baseline_metrics,
        'accuracy_delta_percentage_points': 100 * (native_metrics['accuracy'] - baseline_metrics['accuracy']),
        'paired_correctness': {'both_correct': pairs[(True, True)], 'native_only_correct': pairs[(True, False)],
                               'v6_only_correct': pairs[(False, True)], 'both_wrong': pairs[(False, False)]},
        'native_predictions_sha256': sha256_file(output / 'predictions.jsonl'),
        'baseline_predictions_sha256': sha256_file(old / 'predictions.jsonl'),
        'protocol': 'Same model, labels, image pairs and resize; official AV prompt with verbatim action.description proxy.',
        'interpretation': 'Native prompt omits goal/history and condition generation; task-aware labels are retained. '
                          'This is a prompt/interface transfer comparison, not the authors native benchmark.',
    }
    save_json(output / 'native_vs_v6_comparison.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
