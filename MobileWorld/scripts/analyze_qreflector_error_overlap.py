"""Read existing AndroidWorld predictions; never run inference or change labels."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/analysis/qreflector_error_overlap_20261009'
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def user_text(row):
    return next(x['text'] for x in row['messages'][0]['content'] if x['type'] == 'text')


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_name(path):
    s = str(path)
    leaf = path.parent.name
    if 'qreflector_sft_v6_' in s:
        return 'v6_4bit_' + leaf
    if 'qreflector_bf16_v6_20261004' in s:
        return 'v6_bf16_' + leaf
    if 'v6_continue_' in s:
        return 'v6_bf16_' + leaf
    if 'v6_epoch10_' in s:
        return 'v6_bf16_' + leaf
    if 'opsd_pure_' in s:
        return 'v6_OPSD_pure'
    if 'opsd_v6_' in s:
        return 'v6_OPSD_SFT01'
    if 'qreflector_epoch9_condition_check_' in s:
        return 'v6_epoch9_new_prompt'
    if 'bf16_v8_' in s:
        return 'v8_epoch7' if 'supplemental' in s else 'v8_epoch2'
    if 'bf16_v9_' in s:
        return 'v9_epoch9'
    if 'sol_round1_' in s:
        return 'Sol_round1'
    if 'sol_round2_' in s:
        return 'Sol_round1_geometry' if leaf == 'test_before' else 'Sol_round2_geometry'
    if 'haiku45_' in s:
        return 'Haiku45_v7'
    if 'luna_condition_' in s:
        return 'Luna_v7'
    if 'androidworld1082_native_' in s:
        return 'GUIReflection_native'
    if 'androidworld_1082_' in s:
        return 'GUIReflection_original_shared'
    raise ValueError(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = read_rows(DATA / 'test.jsonl')
    assert len(canonical) == 1082
    labels = {}
    samples = {}
    for i, row in enumerate(canonical, 1):
        t = user_text(row)
        context = json.loads(t.split('=== Context ===', 1)[1].split('=== Image Reference ===', 1)[0])
        target = json.loads(row['messages'][1]['content'][0]['text'])
        labels[i] = target['answer'].lower()
        images = [x['image'] for x in row['messages'][0]['content'] if x['type'] == 'image']
        trajectory = Path(images[0]).name.split('_step_')[0]
        samples[i] = {'sample_id': i, 'source_sample_id': f'test-{i}', 'label': labels[i],
                      'context': context, 'images': images, 'trajectory': trajectory}

    # Verify common test ordering, label semantics, action/context, and image references
    # across the actual frozen dataset versions, rather than trusting numeric IDs alone.
    prompt_hashes = collections.defaultdict(set)
    dataset_checks = []
    for folder in sorted((ROOT / '../datasets').glob('condition_generation_v*')):
        p = folder / 'test.jsonl'
        if not p.exists():
            continue
        rows = read_rows(p)
        assert len(rows) == 1082, p
        changed_context = []
        for i, row in enumerate(rows, 1):
            txt = user_text(row)
            prompt_hashes[i].add(hashlib.sha256(txt.encode()).hexdigest())
            images = [Path(x['image']).name for x in row['messages'][0]['content'] if x['type'] == 'image']
            assert images == [Path(x).name for x in samples[i]['images']], (p, i, 'images')
            assert json.loads(row['messages'][1]['content'][0]['text'])['answer'].lower() == labels[i], (p, i, 'label')
            ctx = json.loads(txt.split('=== Context ===', 1)[1].split('=== Image Reference ===', 1)[0])
            if ctx != samples[i]['context']:
                changed_context.append(i)
        dataset_checks.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                               'image_order_and_labels_match': True, 'changed_context_ids': changed_context})

    import qreflector_geometry_prompt as geo
    for i, row in enumerate(canonical, 1):
        t = user_text(geo.augment_row(row, DATA, 512))
        prompt_hashes[i].add(hashlib.sha256(t.encode()).hexdigest())

    runs, baseline_runs, excluded = {}, {}, []
    for root in ['artifacts/training', 'artifacts/evaluations']:
        for p in sorted((ROOT / root).rglob('predictions.jsonl')):
            rows = read_rows(p)
            if len(rows) != 1082:
                excluded.append({'path': str(p), 'rows': len(rows)})
                continue
            trained = (root.endswith('training') and '/base-qwen' not in str(p)
                       and '/gui-reflection-8b/' not in str(p)) or '/qreflector_epoch9_condition_check_' in str(p)
            by_id = {int(str(r['sample_id']).split('-')[-1]): r for r in rows}
            assert set(by_id) == set(labels), p
            for i, row in by_id.items():
                assert row['expected'] == labels[i], (p, i, 'label')
                assert bool(row['correct']) == (row['predicted'] == labels[i]), (p, i, 'correct')
                assert row.get('source_sample_id', f'test-{i}') == f'test-{i}', (p, i, 'source id')
            matched = sum(r.get('prompt_sha256') in prompt_hashes[i] for i, r in by_id.items())
            if trained:
                assert matched == 1082, (p, 'prompt provenance', matched)
            errors = {i for i, r in by_id.items() if r['predicted'] != labels[i]}
            metrics = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                       'correct': 1082 - len(errors), 'errors': len(errors),
                       'accuracy': (1082-len(errors))/1082,
                       'yes_correct': sum(r['predicted'] == labels[i] == 'yes' for i, r in by_id.items()),
                       'no_correct': sum(r['predicted'] == labels[i] == 'no' for i, r in by_id.items()),
                       'invalid': sum(r['predicted'] not in ('yes', 'no') for r in rows),
                       'matching_frozen_prompt_hashes': matched}
            target = runs if trained else baseline_runs
            name = run_name(p)
            assert name not in target
            target[name] = {'metrics': metrics, 'rows': by_id, 'errors': errors}

    names = list(runs)
    recent_names = ['v9_epoch9', 'Sol_round1', 'Sol_round1_geometry', 'Sol_round2_geometry']
    groups = {'all_24': names, 'recent_v9_sol_4': recent_names,
              'v6_bf16_epochs3_to10': ['v6_bf16_final', 'v6_bf16_step-444', 'v6_bf16_step-555'] +
                 [f'v6_bf16_epoch-{e}-step-{e*111}' for e in range(6, 11)]}
    group_stats = {}
    for name, selected in groups.items():
        freq = {i: sum(i in runs[n]['errors'] for n in selected) for i in labels}
        always = [i for i, f in freq.items() if f == len(selected)]
        never = [i for i, f in freq.items() if f == 0]
        group_stats[name] = {'run_names': selected, 'always_wrong_ids': always,
                            'always_correct_count': len(never), 'ever_wrong_count': 1082-len(never),
                            'always_wrong_labels': dict(collections.Counter(labels[i] for i in always)),
                            'always_wrong_trajectories': dict(collections.Counter(samples[i]['trajectory'] for i in always)),
                            'wrong_count_histogram': dict(sorted(collections.Counter(freq.values()).items()))}

    pairs = []
    for a in names:
        for b in names:
            ea, eb = runs[a]['errors'], runs[b]['errors']
            fixed, regressed = sorted(ea-eb), sorted(eb-ea)
            pairs.append({'from': a, 'to': b, 'common_errors': len(ea&eb),
                          'error_jaccard': len(ea&eb)/len(ea|eb), 'fixed': len(fixed),
                          'regressed': len(regressed), 'net_correct_change': len(fixed)-len(regressed),
                          'fixed_ids': fixed, 'regressed_ids': regressed})

    summary = {'n':1082, 'label_counts':dict(collections.Counter(labels.values())),
               'runs':{n:r['metrics'] for n,r in runs.items()},
               'baselines':{n:r['metrics'] for n,r in baseline_runs.items()}, 'groups':group_stats,
               'recent_sequential_pairs':[next(p for p in pairs if p['from']==a and p['to']==b)
                   for a,b in zip(recent_names,recent_names[1:])],
               'dataset_checks':dataset_checks, 'excluded':excluded,
               'limits':['Runs are correlated checkpoints, not independent replications.',
                         'Prompts, training targets, quantization and schedules vary across settings.',
                         'All errors are disagreements with frozen labels; label errors require independent review.',
                         'Overlap is descriptive and does not establish a model capacity limit or causal mechanism.',
                         'No new inference, teacher API call, training or label change was performed.']}
    write_json('summary.json', summary)
    write_json('pairwise.json', pairs)
    with (OUT/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['run',*next(iter(runs.values()))['metrics']]);w.writeheader()
        for n,r in runs.items():w.writerow({'run':n,**r['metrics']})
    with (OUT/'sample_errors.csv').open('w',newline='') as f:
        fields=['sample_id','label','wrong_of_24','wrong_of_recent4','action_type','trajectory','goal','before','after',*names,*baseline_runs]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,s in samples.items():
            w.writerow({'sample_id':f'test-{i}','label':labels[i],
                'wrong_of_24':sum(i in r['errors'] for r in runs.values()),
                'wrong_of_recent4':sum(i in runs[n]['errors'] for n in recent_names),
                'action_type':s['context']['action']['type'],'trajectory':s['trajectory'],
                'goal':s['context']['goal'],'before':str(DATA/s['images'][0]),'after':str(DATA/s['images'][1]),
                **{n:r['rows'][i]['predicted'] or 'invalid' for n,r in {**runs,**baseline_runs}.items()}})
    # Preserve every shared failure for screenshot-based follow-up without opening a GPU model.
    shared_ids=group_stats['recent_v9_sol_4']['always_wrong_ids']
    core_ids=group_stats['all_24']['always_wrong_ids']
    with (OUT/'persistent_cases.jsonl').open('w') as f:
        for i in sorted(set(shared_ids)|set(core_ids)):
            record={**samples[i], 'wrong_of_24':sum(i in r['errors'] for r in runs.values()),
                    'wrong_of_recent4':sum(i in runs[n]['errors'] for n in recent_names),
                    'predictions':{n:r['rows'][i] for n,r in {**runs,**baseline_runs}.items()}}
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps({'runs':len(runs),'baselines':len(baseline_runs),
                      'all_wrong':len(core_ids),'recent_all_wrong':len(shared_ids),
                      'output':str(OUT)}))


if __name__ == '__main__':
    main()
