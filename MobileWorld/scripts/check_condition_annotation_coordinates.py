#!/usr/bin/env python3
"""Correct annotation coordinate-frame confusion; never changes student inputs."""
from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import package_condition_check_data as package
import qreflector_condition_check_data as data
from PIL import Image
from refine_condition_check_annotations import REVIEW_INSTRUCTION, send


def map_coordinates(width, height, max_side, args):
    factor = min(1, max_side / max(width, height))
    resized = [int(width * factor), int(height * factor)]
    scaled = {}
    for key in ('x', 'y', 'x1', 'y1', 'x2', 'y2'):
        value = args.get(key)
        if isinstance(value, (int, float)):
            scaled[key] = round(value * resized[0 if key.startswith('x') else 1]
                                / (width if key.startswith('x') else height), 2)
    return {'original_screen_width_height': [width, height],
            'uploaded_image_width_height': resized, 'original_action_arguments': args,
            'equivalent_coordinates_on_uploaded_image': scaled}


def coordinate_hint(row, user, plan):
    prefix, suffix = (data.OUTPUT / 'prompt_template.txt').read_text().split('{context}')
    context = json.loads(user['content'][0]['text'][len(prefix):-len(suffix)])
    images = [c['image'] for c in row['messages'][0]['content'] if c['type'] == 'image']
    with Image.open(data.DATA / images[0]) as im:
        width, height = im.size
    args = context['action'].get('arguments') or {}
    return ('\nCOORDINATE FRAME (verified image metadata, not a label):\n'
            + json.dumps(map_coordinates(width, height, plan['image_max_side'], args))
            + '\nAll action coordinates and coordinates quoted in conditions refer to the ORIGINAL '
              'screen, not the resized upload. Compare them using this supplied mapping. Do not '
              'call a coordinate out-of-bounds by comparing it against the resized width/height. '
              'Do not include preprocessing details in the student training reason.\n')


def run_one(output, plan, key, split, line, row, user, source, annotation):
    sample_id = f'{split}-{line}'
    revised = copy.deepcopy(row)
    revised['messages'][1]['content'][0]['text'] = json.dumps(source)
    payload, hashes = data.build_request(revised, user, plan)
    payload['max_output_tokens'] = 6144
    payload['input'][0]['content'][0]['text'] += coordinate_hint(row, user, plan)
    payload['input'][0]['content'][0]['text'] += REVIEW_INSTRUCTION
    if annotation:
        payload['input'][0]['content'][0]['text'] += '\nUNTRUSTED ANNOTATION TO CHECK:\n' + json.dumps(annotation)
    return send(payload, output / 'coordinate_review/rechecks' / f'{sample_id}.json', key,
                {'sample_id': sample_id, 'stage': 'evidence_recheck',
                 'conditions': {k: source[k] for k in ('precondition', 'postcondition')},
                 'submitted_image_sha256': hashes, 'original_label_sent': False,
                 'original_reason_sent': False, 'explicit_original_coordinate_mapping': True})


def main():
    output = data.OUTPUT
    plan, key = data.verify_plan(output), data.read_api_key(data.ROOT / '.env')
    (output / 'coordinate_review/rechecks').mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ('train', 'val'):
        for line, (row, inp) in enumerate(zip(data.read_rows(data.DATA / f'{split}.jsonl'),
                                            data.read_rows(output / f'{split}.inputs.jsonl'), strict=True), 1):
            source = data.target(row)
            candidate, audit = package.select_annotation(output, f'{split}-{line}', source)
            a = audit.get('annotation')
            # Include all reasons discussing coordinate alignment, even when their
            # verdict happens to agree with the label. Selection is not verdict-only.
            text = json.dumps(a or {})
            if re.search(r'coordinat|tap (?:point|location)|tapped location|outside.*screen|out.of.bounds|mapped tap', text, re.I) or 'invalid_refinement' in audit['flags']:
                jobs.append((output, plan, key, split, line, row, inp['messages'][0],
                             candidate or source, a))
    data.write_json(output / 'coordinate_review/plan.json', {
        'created_utc': data.now(), 'rows': len(jobs), 'max_output_tokens': 6144,
        'selection': 'All latest annotations mentioning coordinate alignment or an incomplete recheck, '
                     'including label-agreeing rows.',
        'original_label_or_reason_sent': False, 'test_rows_sent': 0,
        'changes_student_inputs': False, 'changes_source_images_or_actions': False})
    print(json.dumps({'coordinate_review_rows': len(jobs)}), flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, future in enumerate(as_completed([pool.submit(run_one, *job) for job in jobs]), 1):
            result = future.result()
            if i % 25 == 0 or i == len(jobs):
                print(json.dumps({'coordinate_reviews_completed': i, 'total': len(jobs),
                                  'last': result['sample_id']}), flush=True)


if __name__ == '__main__':
    main()
