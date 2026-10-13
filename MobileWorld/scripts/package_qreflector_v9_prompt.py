#!/usr/bin/env python3
"""Freeze the revised prompt with unchanged v8 examples, labels and targets."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from eval_qreflector_prompt import render_user
from qreflector_sft_v8 import DATA as SOURCE, ROOT, digest, read_rows, validate_dataset

DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
PROMPT = ROOT / 'src/mobile_world/reflection/prompts/with_condition_generation.txt'
SPLITS = {'train': 880, 'val': 71, 'test': 1082}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def build(output):
    assert output.parent.resolve() == SOURCE.parent.resolve() and output != SOURCE
    source_manifest = validate_dataset(SOURCE, splits=tuple(SPLITS), images=True)
    replacement = PROMPT.read_text()
    source_prompt = (SOURCE / 'prompt_template.txt').read_text()
    output.mkdir(exist_ok=False)
    (output / 'images').symlink_to((SOURCE / 'images').resolve(), target_is_directory=True)
    (output / 'image_sha256.json').symlink_to((SOURCE / 'image_sha256.json').resolve())
    (output / 'prompt_template.txt').write_text(replacement)
    shutil.copyfile(PROMPT.with_suffix('.metadata.json'), output / 'prompt_metadata.snapshot.json')
    shutil.copyfile(SOURCE / 'quarantine.jsonl', output / 'quarantine.jsonl')
    changes = {}
    for split, count in SPLITS.items():
        original_rows = read_rows(SOURCE / f'{split}.jsonl')
        labels = read_rows(SOURCE / f'{split}.labels.jsonl')
        assert len(original_rows) == len(labels) == count
        rows, inputs, input_manifest = [], [], []
        for line, (original, label) in enumerate(zip(original_rows, labels, strict=True), 1):
            user, meta = render_user(original, source_prompt, replacement)
            row = copy.deepcopy(original)
            row['messages'][0] = user
            assert row['messages'][1:] == original['messages'][1:]
            assert user['content'][1:] == original['messages'][0]['content'][1:]
            assert json.loads(row['messages'][1]['content'][0]['text'])['answer'] == label['answer']
            rows.append(row)
            inputs.append({'messages': [user]})
            input_manifest.append({'sample_id': label['sample_id'], 'source_line': line, **meta})
        write_rows(output / f'{split}.jsonl', rows)
        write_rows(output / f'{split}.inputs.jsonl', inputs)
        write_rows(output / f'{split}.input_manifest.jsonl', input_manifest)
        shutil.copyfile(SOURCE / f'{split}.labels.jsonl', output / f'{split}.labels.jsonl')
        line_map = SOURCE / f'{split}.line_map.jsonl'
        if line_map.exists():
            shutil.copyfile(line_map, output / line_map.name)
        reference = SOURCE / (f'{split}.references.jsonl' if split != 'test' else 'test.source_references.jsonl')
        (output / f'{split}.source_references.jsonl').symlink_to(reference.resolve())
        changes[split] = {'rows': count, 'contexts_unchanged': True, 'history_unchanged': True,
                         'ordered_images_unchanged': True, 'assistant_targets_unchanged': True,
                         'labels_unchanged': True, 'changed_user_prompts': count}
    source_pins = {'DATASET_MANIFEST.json': digest(SOURCE / 'DATASET_MANIFEST.json'),
                   **source_manifest['artifact_sha256']}
    for name, expected in source_pins.items():
        assert digest(SOURCE / name) == expected, f'Source mutated: {name}'
    (output / 'README.txt').write_text(
        'Q-Reflector v9: before-derived expectations, after-based verification.\n'
        'Only the user prompt is changed from v8. All 880 training, 71 validation,\n'
        'and 1082 test examples retain their context, image order, labels and full\n'
        'assistant targets. No teacher annotation or relabeling is performed.\n'
        'Images are referenced by symlink, not copied. Source data is preserved.\n'
        'The existing targets inherit v8 limitations; new wording does not prove\n'
        'that every condition was generated independently of AFTER.\n'
        'The prompt was revised after examining existing test errors. Report a\n'
        'subsequent test result as exploratory, not an untouched blind holdout.\n'
        'Checkpoint selection uses final Yes/No accuracy on validation only.\n'
    )
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'frozen',
        'dataset_version': output.name, 'source': str(SOURCE),
        'source_manifest_sha256': source_pins['DATASET_MANIFEST.json'],
        'source_artifact_sha256': source_pins,
        'splits': copy.deepcopy(source_manifest['splits']), 'test_rows': 1082,
        'quarantined_ids': ['train-243', 'train-589'],
        'prompt_source': str(PROMPT), 'prompt_sha256': digest(PROMPT),
        'changes': changes, 'images_copied': 0, 'source_images_verified': True,
        'assistant_targets_changed': False, 'labels_changed': False,
        'test_used_for_checkpoint_selection': False,
        'prompt_development_used_previous_test_error_analysis': True,
        'packager_sha256': digest(Path(__file__)),
        'artifact_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    (output / 'DATASET_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'dataset': str(output), 'changes': changes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA)
    build(parser.parse_args().output.resolve())
