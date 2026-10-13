#!/usr/bin/env python3
"""Audit paired correction artifacts and account for cache-write token charges."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import qreflector_teacher_pilot_v9 as pilot
import qreflector_luna_correction as correction
import qreflector_opsd as shared
from qreflector_sft_bf16 import read_rows

RATES = {
    'gpt-6-luna': {'input': 0.10, 'cached_input': 0.01, 'cache_write': 0.125, 'output': 0.50},
    'gpt-6.1-sol': {'input': 2.00, 'cached_input': 0.10, 'cache_write': 2.50, 'output': 10.00},
}


def cost(events, model):
    counts = Counter()
    for event in events:
        usage = event.get('usage') or {}
        details = usage.get('input_tokens_details') or {}
        inp = usage.get('input_tokens', 0)
        cached, writes = details.get('cached_tokens', 0), details.get('cache_write_tokens', 0)
        assert cached + writes <= inp
        counts['input_tokens'] += inp
        counts['uncached_nonwrite_tokens'] += inp - cached - writes
        counts['cached_input_tokens'] += cached
        counts['cache_write_tokens'] += writes
        counts['output_tokens'] += usage.get('output_tokens', 0)
        counts['reasoning_tokens'] += (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    rates = RATES[model]
    value = (counts['uncached_nonwrite_tokens'] * rates['input'] +
             counts['cached_input_tokens'] * rates['cached_input'] +
             counts['cache_write_tokens'] * rates['cache_write'] +
             counts['output_tokens'] * rates['output']) / 1e6
    return {**counts, 'estimated_standard_api_cost_usd': value,
            'rates_usd_per_million': rates,
            'pricing_source': f'https://developers.openai.com/api/docs/models/{model}',
            'pricing_checked_utc_date': '2026-10-08',
            'includes_reasoning_and_cache_writes': True}


def summarize(run, verify_requests=False):
    plan, samples = pilot.load_plan(run)
    drafts = read_rows(run / 'student_rollouts.jsonl')
    assert len(drafts) == 100
    for sample, draft in zip(samples, drafts):
        assert draft['sample_id'] == sample['sample_id']
        assert draft['student_messages_sha256'] == sample['student_messages_sha256']
    output = {'complete': True, 'base': plan['base_description'], 'samples': 100,
              'train_only': True, 'weight_updates': False, 'student_sampling': plan['student'],
              'student_label_agreement': sum(d['predicted'] == d['expected'].lower() for d in drafts),
              'student_valid_four_field_json': sum(d['valid_four_field_json'] for d in drafts),
              'models': {}, 'request_pairs_verified': verify_requests,
              'caveat': 'Teacher sees train references. Agreement is not independent test accuracy. Automatic eligibility is not a semantic quality guarantee.'}
    correction.TEACHER_INSTRUCTIONS = plan['teacher_instructions']
    exclusions_file = run / 'manual_exclusions.json'
    exclusions = shared.read_json(exclusions_file) if exclusions_file.exists() else {}
    by_model = {}
    for model in pilot.TEACHERS:
        records = read_rows(run / model / 'corrections.jsonl')
        events = read_rows(run / model / 'api_requests.jsonl')
        assert len(records) == 100
        assert len({r['request_sha256'] for r in records}) == 100
        counts = Counter()
        candidates = []
        for i, (sample, draft, record) in enumerate(zip(samples, drafts, records)):
            assert record['sample_id'] == sample['sample_id']
            assert record['draft_sha256'] == correction.digest_json(draft['full_response'])
            assert record['teacher_model'] == model
            if verify_requests:
                cfg = {'data': plan['data'], 'image_max_side': 512, 'teacher_model': model,
                       'teacher_reasoning': 'medium', 'teacher_max_output_tokens': 4096}
                request = correction.teacher_request(sample['row'], draft['full_response'], cfg)
                assert correction.digest_json(request) == record['request_sha256']
            target = (record.get('parsed') or {}).get('corrected_response') or {}
            verdict = target.get('answer')
            counts['label_agreement'] += verdict == sample['reference']['answer']
            counts['automatically_eligible'] += record['accepted']
            counts['fixed_student_label'] += draft['predicted'] != draft['expected'].lower() and verdict == draft['expected']
            counts['changed_correct_student_label'] += draft['predicted'] == draft['expected'].lower() and verdict != draft['expected']
            checks = all(k in target.get('reason', '') for k in ('Precondition check:', 'Postcondition check:', 'Task progress:'))
            counts['three_reason_checks'] += checks
            manually_excluded = sample['sample_id'] in exclusions.get(model, {})
            counts['additional_manual_exclusions_after_automatic_filter'] += record['accepted'] and manually_excluded
            if record['accepted'] and checks and not manually_excluded:
                messages = correction.student_messages(sample['row'])
                messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(target, ensure_ascii=False)}]})
                candidates.append({'sample_id': sample['sample_id'], 'messages': messages,
                    'original_data_root': plan['data'], 'teacher_request_sha256': record['request_sha256'],
                    'semantic_review_required': True})
        (run / model / 'candidate_targets.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in candidates))
        output['models'][model] = {**counts, 'candidate_targets_saved': len(candidates),
            'manual_exclusions': exclusions.get(model, {}),
            'decisions': dict(Counter(r['decision'] for r in records)),
            'api_calls': sum(e['event'] == 'request_started' for e in events),
            'response_status': dict(Counter(r['status'] for r in records)),
            'latency_mean_seconds': statistics.mean(r['elapsed_seconds'] for r in records),
            'latency_median_seconds': statistics.median(r['elapsed_seconds'] for r in records),
            'cost': cost(events, model)}
        by_model[model] = records
    disagreements = []
    for a,b in zip(by_model[pilot.TEACHERS[0]], by_model[pilot.TEACHERS[1]]):
        assert a['sample_id'] == b['sample_id'] and a['draft_sha256'] == b['draft_sha256']
        av = (a.get('parsed') or {}).get('corrected_response', {}).get('answer')
        bv = (b.get('parsed') or {}).get('corrected_response', {}).get('answer')
        if av != bv or not a['accepted'] or not b['accepted']:
            disagreements.append({'sample_id':a['sample_id'], 'luna_answer': av, 'sol_answer':bv,
                                  'luna_decision':a['decision'], 'sol_decision':b['decision']})
    output['disagreements_and_flags'] = disagreements
    shared.write_json(run / 'comparison.json', output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-requests', action='store_true')
    args = p.parse_args()
    summarize(args.output, args.verify_requests)
