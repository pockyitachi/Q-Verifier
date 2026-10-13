#!/usr/bin/env python3
"""Package audited condition-check data; gate unresolved labels, preserve v6."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import qreflector_condition_check_data as data


def select_annotation(output, sample_id, source):
    first = json.loads((output / 'annotations' / f'{sample_id}.json').read_text())
    annotation, conditions, stage = first['annotation'], {k: source[k] for k in data.FIELDS[:2]}, 'first_pass'
    record = first
    flags = []
    for directory in ('refinements', 'refinements_before_only', 'coordinate_review'):
        path = output / directory / 'rechecks' / f'{sample_id}.json'
        if path.exists():
            second = json.loads(path.read_text())
            if second['value'] is not None:
                record, stage = second, directory
                annotation, conditions = second['value'], second['conditions']
                flags = []
            else:
                flags = ['invalid_refinement']
    if annotation is None:
        flags.append('invalid_annotation')
        return None, {'flags': flags, 'stage': stage, 'response_id': record['response_id']}
    candidate = data.assemble_target({**source, **conditions}, annotation)
    if annotation['evidence_verdict'] != source['answer']:
        flags.append('label_evidence_disagreement')
    if not annotation['conditions_suitable']:
        flags.append('condition_semantics_issue')
    return candidate, {'flags': flags, 'stage': stage, 'response_id': record['response_id'],
                       'annotation': annotation, 'conditions': conditions}


def package(output):
    plan = data.verify_plan(output)
    data.validate_dataset(data.DATA, images=True)
    decisions_path = output / 'review_decisions.jsonl'
    decisions = {r['sample_id']: r for r in data.read_rows(decisions_path)} if decisions_path.exists() else {}
    summaries, audits, references, all_examples = {}, [], {}, []
    changed = Counter()
    from tokenizers import Tokenizer
    tokenizer_path = data.ROOT / 'artifacts/models/qreflector_base_bf16/Qwen3-VL-8B-Instruct/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_lengths = []
    for split in ('train', 'val'):
        source_rows = data.read_rows(data.DATA / f'{split}.jsonl')
        inputs = data.read_rows(output / f'{split}.inputs.jsonl')
        candidates, consistent, mapping, refs = [], [], [], []
        for line, (row, inp) in enumerate(zip(source_rows, inputs, strict=True), 1):
            sample_id, source = f'{split}-{line}', data.target(row)
            value, audit = select_annotation(output, sample_id, source)
            if value is None:
                raise ValueError(f'{sample_id}: missing a structurally valid target; do not fill with old prose')
            audit.update({'sample_id': sample_id, 'source_line': line, 'split': split,
                          'source_label': source['answer']})
            if sample_id in decisions:
                decision = decisions[sample_id]
                if decision.get('source_target_sha256') != hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest():
                    raise ValueError(f'Stale review decision: {sample_id}')
                replacement = decision.get('target')
                if replacement:
                    if replacement['answer'] != source['answer']:
                        raise ValueError('Review cannot relabel source data')
                    value = copy.deepcopy(replacement)
                if decision['status'] == 'resolved':
                    audit['automatic_flags_before_review'] = audit['flags']
                    audit['flags'] = []
                audit['review'] = decision
            if list(value) != list(data.FIELDS):
                raise ValueError(f'{sample_id}: wrong field order')
            if any(not isinstance(v, str) or not v.strip() for v in value.values()):
                raise ValueError(f'{sample_id}: empty target')
            if re.search(r'[\u3400-\u9fff]', ' '.join(value.values())):
                raise ValueError(f'{sample_id}: non-English target')
            reason = value['reason']
            if (not reason.startswith(data.HEADINGS[0])
                    or any(reason.count(h) != 1 for h in data.HEADINGS)
                    or reason.index(data.HEADINGS[1]) > reason.index(data.HEADINGS[2])):
                raise ValueError(f'{sample_id}: incorrect reason structure')
            assert value['answer'] == source['answer']
            rendered, meta = data.render_user(row, (data.DATA / 'prompt_template.txt').read_text(),
                                               (output / 'prompt_template.txt').read_text())
            assert rendered == inp['messages'][0]
            for part in rendered['content'][1:]:
                assert (output / part['image']).resolve() == (data.DATA / part['image']).resolve()
            serialized = json.dumps(value, ensure_ascii=False)
            n_tokens = len(tokenizer.encode(serialized, add_special_tokens=False).ids)
            token_lengths.append(n_tokens)
            if n_tokens >= 1024:
                audit['flags'].append('target_exceeds_current_1024_token_generation_budget')
            candidate = {'messages': [rendered, {'role': 'assistant',
                         'content': [{'type': 'text', 'text': serialized}]}]}
            candidates.append(candidate)
            audit['changed_fields'] = [key for key in data.FIELDS if value[key] != source[key]]
            audit['target_tokens'] = n_tokens
            changed.update(audit['changed_fields'])
            if not audit['flags']:
                consistent.append(candidate)
                mapping.append({'output_line': len(consistent), 'source_line': line, 'sample_id': sample_id})
            reference = {'sample_id': sample_id, 'source_file': str(data.DATA / f'{split}.jsonl'),
                         'source_line': line, 'original_target': source, 'candidate_target': value,
                         'image_paths': meta['image_paths'], 'context_sha256': meta['context_sha256'],
                         'audit': audit}
            prefix, suffix = (output / 'prompt_template.txt').read_text().split('{context}')
            reference['context'] = json.loads(rendered['content'][0]['text'][len(prefix):-len(suffix)])
            refs.append(reference)
            audits.append(audit)
            if sample_id in ('train-1', 'train-2', 'train-138', 'train-400', 'train-743', 'train-877', 'val-1'):
                all_examples.append(reference)
        data.write_rows(output / f'{split}.candidates.jsonl', candidates)
        data.write_rows(output / f'{split}.ready.jsonl', consistent)
        data.write_rows(output / f'{split}.ready.line_map.jsonl', mapping)
        data.write_rows(output / f'{split}.references.jsonl', refs)
        references[split] = refs
        summaries[split] = {'source_rows': len(source_rows), 'rewritten_rows': len(candidates),
                            'consistent_rows': len(consistent), 'pending_rows': len(source_rows) - len(consistent),
                            'source_labels': dict(Counter(data.target(r)['answer'] for r in source_rows)),
                            'candidate_labels': dict(Counter(data.target(r)['answer'] for r in candidates)),
                            'consistent_labels': dict(Counter(data.target(r)['answer'] for r in consistent))}
    pending = [a for a in audits if a['flags']]
    data.write_rows(output / 'annotation_audit.jsonl', audits)
    data.write_rows(output / 'review_required.jsonl', [
        r for refs in references.values() for r in refs if r['audit']['flags']])
    data.write_rows(output / 'examples.before_after.jsonl', all_examples)
    source_test, revised_test = (data.read_rows(p / 'test.jsonl') for p in (data.DATA, output))
    assert len(source_test) == len(revised_test) == 1082
    assert all(a['messages'][1] == b['messages'][1] for a, b in zip(source_test, revised_test, strict=True))
    assert len(list((output / 'annotations').glob('test-*.json'))) == 0
    report = {'created_utc': data.now(), 'dataset_version': output.name,
              'status': 'draft_pending_adjudication' if pending else 'ready_not_frozen',
              'source_version': data.DATA.name, 'prompt_sha256': plan['prompt_sha256'],
              'splits': summaries, 'test': {'rows': 1082, 'input_prompt_updated': True,
                                         'labels_and_legacy_assistant_reference_unchanged': True,
                                         'sent_for_annotation': 0},
              'changes': dict(changed), 'pending_rows': len(pending),
              'pending_flags': dict(Counter(f for a in pending for f in a['flags'])),
              'annotation_rows_complete': 953, 'labels_changed': 0, 'images_copied': 0,
              'original_frozen_files_and_images_verified': True,
              'maximum_target_text_tokens': max(token_lengths),
              'mean_target_text_tokens': sum(token_lengths) / len(token_lengths),
              'training_started': False, 'gpu_used': False,
              'full_882_training_ready': summaries['train']['pending_rows'] == 0,
              'training_ready': not pending,
              'limitations': [
                  'API-assisted annotations and consistency checks are not full human validation.',
                  'Candidate targets keep original labels even on disagreements; do not train on pending rows.',
                  'The ready subset is NOT the original 882/71 split; do not silently substitute it in comparisons.',
                  'Test legacy reasons are preserved for provenance/scoring, not SFT or validation loss.',
                  'Annotation/refinement saw up to 1280px screenshots; student evaluation has used 512px.',
                  'Text token checks are not the later multimodal collator preflight.',
                  'Changed supervision as well as changed prompt: this is not a pure prompt ablation.']}
    data.write_json(output / 'conversion_report.json', report)
    artifacts = [p for p in output.iterdir() if p.is_file() and not p.is_symlink()
                 and p.name not in ('DATASET_MANIFEST.json', 'status.json', 'api_requests.jsonl', '.annotation.lock')]
    data.write_json(output / 'DATASET_MANIFEST.json', {**report,
        'artifact_sha256': {p.name: data.digest(p) for p in artifacts},
        'source_frozen_sha256': data.digest(data.DATA / 'FROZEN.json'),
        'image_manifest_source': str(data.DATA / 'image_sha256.json')})
    data.write_json(output / 'status.json', {'status': report['status'], 'updated_utc': data.now(),
                    'annotation_rows_complete': 953, 'pending_rows': len(pending), 'training_started': False})
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=data.OUTPUT)
    args = parser.parse_args()
    package(args.output.resolve())


if __name__ == '__main__':
    main()
