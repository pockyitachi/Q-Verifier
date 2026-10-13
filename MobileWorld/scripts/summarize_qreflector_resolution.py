#!/usr/bin/env python3
"""Analyze completed, fixed-checkpoint validation resolution runs; no inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics
import qreflector_geometry_prompt as geometry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/evaluations/qreflector_round2_val_resolution_20261009'
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    source_rows = rows(DATA / 'val.jsonl')
    data = {}
    metric_rows = []
    answer_only = {}
    protected = {}
    for side in (512, 1024):
        folder = OUT / f'side_{side}'
        state = read(folder / 'status.json')
        assert state['status'] == 'complete' and state['completed'] == 71, state
        manifest = read(folder / 'manifest.json')
        predictions = rows(folder / 'predictions.jsonl')
        summary = read(folder / 'summary.json')
        assert len(predictions) == 71
        assert [r['sample_id'] for r in predictions] == list(range(1, 72))
        assert summary['total'] == 71 and summary['correct'] == sum(r['correct'] for r in predictions)
        for p, h in manifest['pinned_files'].items():
            assert sha(Path(p)) == h, p
        data[side] = (manifest, predictions)
        final_answers = []
        for rec in predictions:
            matches = re.findall(r'"answer"\s*:\s*"(Yes|No)"\s*}\s*$', rec['full_response'])
            verdict = matches[0].lower() if len(matches) == 1 else None
            if rec['predicted'] is not None:
                assert verdict == rec['predicted']
            final_answers.append(verdict)
        answer_only[side] = final_answers
        for name in ('status.json', 'manifest.json', 'predictions.jsonl', 'summary.json', 'model_validation.json'):
            protected[str(folder / name)] = sha(folder / name)
        metric_rows.append({
            'long_side': side,
            'correct': summary['correct'],
            'total': 71,
            'accuracy_pct': 100 * summary['accuracy'],
            'answer_only_correct': sum(v == rec['expected'] for v, rec in zip(final_answers, predictions)),
            'answer_only_accuracy_pct': 100 * sum(v == rec['expected'] for v, rec in zip(final_answers, predictions)) / 71,
            'success_correct': summary['confusion']['yes']['yes'],
            'success_total': summary['success_total'],
            'success_recognition_pct': 100 * summary['success_recognition_rate'],
            'failure_correct': summary['confusion']['no']['no'],
            'failure_total': summary['failure_total'],
            'failure_recognition_pct': 100 * summary['failure_recognition_rate'],
            'invalid_answers': summary['invalid_outputs'],
            'invalid_structure': sum(not r['valid_four_field_json'] for r in predictions),
            'generation_limit_hits': sum(r['at_generation_limit'] for r in predictions),
            'mean_input_tokens': statistics.mean(r['input_tokens'] for r in predictions),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in predictions),
            'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in predictions),
            'median_latency_seconds': statistics.median(r['latency_seconds'] for r in predictions),
            'evaluation_seconds': state['elapsed_seconds'],
            'peak_allocated_gib': state['peak_allocated_gib'],
        })
    ma, pa = data[512]
    mb, pb = data[1024]
    differing_manifest = {k for k in ma if ma[k] != mb[k]}
    assert differing_manifest <= {'image_max_side', 'gpu_uuid', 'pid', 'started_at_unix'}
    assert ma['pinned_files'] == mb['pinned_files']
    fixed, regressed, changes = [], [], []
    both_wrong, both_correct = [], []
    image_hashes = {}
    for source, a, b in zip(source_rows, pa, pb):
        assert a['source_sample_id'] == b['source_sample_id']
        assert a['expected'] == b['expected']
        assert a['context_sha256'] == b['context_sha256']
        assert a['prompt_sha256'] != b['prompt_sha256']
        # The dynamic geometry text changes with image dimensions by design.
        assert a['image_grid_thw'] != b['image_grid_thw']
        assert a['image_geometry']['current_action_points'] == b['image_geometry']['current_action_points']
        for side, rec in ((512, a), (1024, b)):
            augmented = geometry.augment_row(source, DATA, side)
            text = augmented['messages'][0]['content'][0]['text']
            assert hashlib.sha256(text.encode()).hexdigest() == rec['prompt_sha256']
            assert text.replace(geometry.geometry_block(source, DATA, side), '', 1) == source['messages'][0]['content'][0]['text']
        for part in source['messages'][0]['content']:
            if part['type'] == 'image':
                path = (DATA / part['image']).resolve()
                image_hashes[str(path)] = sha(path)
        assert len(a['image_geometry']['images']) == len(b['image_geometry']['images']) == 2
        for ai, bi in zip(a['image_geometry']['images'], b['image_geometry']['images']):
            for key in ('original_width', 'original_height'):
                assert ai[key] == bi[key]
        if b['correct'] and not a['correct']:
            fixed.append(a['sample_id'])
        elif a['correct'] and not b['correct']:
            regressed.append(a['sample_id'])
        elif a['correct'] and b['correct']:
            both_correct.append(a['sample_id'])
        else:
            both_wrong.append(a['sample_id'])
        if a['predicted'] != b['predicted']:
            changes.append({'sample_id': a['sample_id'], 'expected': a['expected'],
                'side_512': a, 'side_1024': b,
                'change': 'fixed' if b['correct'] else 'regressed'})
    discordant = len(fixed) + len(regressed)
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(fixed), len(regressed)) + 1)) / 2**discordant) if discordant else 1.
    result = {
        'split': 'validation', 'n': 71,
        'checkpoint': ma['adapter'],
        'comparison': '512 vs 1024 image long side with matching dynamic geometry dimensions; weights/prompt template/context/decoding fixed.',
        'metrics': metric_rows,
        'fixed_ids': fixed, 'regressed_ids': regressed,
        'both_wrong_ids': both_wrong, 'both_correct_ids': both_correct,
        'net_correct_change': len(fixed) - len(regressed),
        'delta_accuracy_pp': 100 * (len(fixed) - len(regressed)) / 71,
        'mcnemar_exact_two_sided_p': p,
        'answer_only_supplement': {
            'policy': 'Same anchored terminal answer-field extraction for all 142 responses; unparseable or ambiguous fields remain incorrect. Original parser predictions and headline strict scores are unchanged.',
            'reason': 'Separate Yes/No judgment accuracy from malformed JSON in intermediate text, consistent with the user interest in final judgments.',
            'cases_recovered': {str(side): [rec['sample_id'] for v, rec in zip(answer_only[side], data[side][1]) if rec['predicted'] is None and v is not None] for side in (512, 1024)},
            'fixed_ids': [i+1 for i in range(71) if answer_only[512][i] != pa[i]['expected'] and answer_only[1024][i] == pa[i]['expected']],
            'regressed_ids': [i+1 for i in range(71) if answer_only[512][i] == pa[i]['expected'] and answer_only[1024][i] != pa[i]['expected']],
        },
        'limitations': [
            '71 validation cases; exploratory estimate, not an independent 1082-case test result.',
            'This existing validation set has been reused in previous model development.',
            'Different idle GPUs ran the two settings; latency is descriptive, not a controlled hardware benchmark.',
            'Image pixels and corresponding dynamic geometry dimensions both change; this is a deployment resolution comparison.',
            'Answer flips alone do not establish improved OCR or causal mechanisms; explanation evidence should be inspected.',
            'No retraining, test inference, or original label edits were performed by this ablation.',
        ],
        'verification': {
            'identical_pinned_sources': True, 'same_context_and_labels': True,
            'all_image_token_grids_changed': True,
            'full_generated_prompts_reconstructed_and_verified': True,
            'dynamic_action_coordinates_identical': True,
            'image_hashes_at_analysis': image_hashes,
            'differing_manifest_keys': sorted(differing_manifest),
            'source_hashes': protected,
        },
    }
    write(OUT / 'comparison.json', result)
    write(OUT / 'changed_cases.json', changes)
    with (OUT / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    report = [
        'Fixed round-2 checkpoint: validation resolution comparison', '',
        f'Checkpoint: {ma["adapter"]}',
        'Same BF16 weights, v9 prompt template, contexts, 71 original labels and greedy generation.',
        'Only image resolution and matching dynamic geometry dimensions change.', '',
        'long side | correct | accuracy | success recognition | failure recognition | mean seconds/case',
    ]
    for r in metric_rows:
        report.append(f'{r["long_side"]} | {r["correct"]}/71 | {r["accuracy_pct"]:.2f}% | {r["success_correct"]}/{r["success_total"]} ({r["success_recognition_pct"]:.2f}%) | {r["failure_correct"]}/{r["failure_total"]} ({r["failure_recognition_pct"]:.2f}%) | {r["mean_latency_seconds"]:.2f}')
    report += ['', f'Fixed at 1024: {fixed}', f'New errors at 1024: {regressed}',
        f'Wrong in both: {both_wrong}',
        f'Net accuracy change: {result["delta_accuracy_pp"]:+.2f} percentage points.',
        f'Exact paired McNemar two-sided p: {p:.6f}', '',
        'Supplement: terminal Yes/No extraction applied uniformly to both runs (original scores retained).']
    report += [f'{r["long_side"]}: {r["answer_only_correct"]}/71 = {r["answer_only_accuracy_pct"]:.2f}%' for r in metric_rows]
    report += ['At 1024, case 68 ends with the correct answer No, but unescaped quotation marks in precondition make the JSON invalid.',
        'Qualitative review: qualitative_review.json. Some changes are task-progress criteria, not demonstrably better visual grounding.', '', *result['limitations']]
    (OUT / 'REPORT.txt').write_text('\n'.join(report) + '\n')
    print(json.dumps({k: result[k] for k in ['metrics', 'fixed_ids', 'regressed_ids', 'both_wrong_ids', 'mcnemar_exact_two_sided_p']}, indent=2))


if __name__ == '__main__':
    main()
