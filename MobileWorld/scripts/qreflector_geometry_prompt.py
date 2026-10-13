"""Add deterministic image geometry to the existing v9 user input.

Uses image dimensions and current action coordinates only. Never reads target
answers or image pixels. The original action and screenshot paths are retained.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from PIL import Image

MARKER = '=== Image Geometry ==='
END_MARKER = '=== End Image Geometry ==='
INSTRUCTIONS = '''The images may be resized. Current-action coordinates are pixels in the
original BEFORE screenshot, with origin at top-left, x increasing rightward
and y increasing downward. Normalized coordinates are (x / original_width,
y / original_height): (0, 0) is top-left and (1, 1) is bottom-right.
Use the normalized positions to locate action targets regardless of resizing.
Do not interpret original-image pixel coordinates as resized-image pixels.
This is geometric metadata, not evidence of success or the identity of a control.
'''


def context_of(row):
    text = next(x['text'] for x in row['messages'][0]['content'] if x['type'] == 'text')
    return json.loads(text.split('=== Context ===', 1)[1].split('=== Image Reference ===', 1)[0])


def geometry(row, data, max_side=512):
    images = []
    for item in row['messages'][0]['content']:
        if item['type'] != 'image':
            continue
        with Image.open(Path(data) / item['image']) as picture:
            w, h = picture.size
        scale = min(1., max_side / max(w, h))
        images.append({'role': 'BEFORE' if not images else 'AFTER',
                       'original_width': w, 'original_height': h,
                       'input_width': int(w * scale), 'input_height': int(h * scale)})
    if len(images) != 2:
        raise ValueError('Exactly two ordered screenshots are required')
    before = images[0]
    args = context_of(row)['action'].get('arguments') or {}
    points = []
    for xk, yk in [('x', 'y'), ('x1', 'y1'), ('x2', 'y2')]:
        if xk not in args and yk not in args:
            continue
        if xk not in args or yk not in args:
            raise ValueError('Incomplete coordinate pair')
        x, y = args[xk], args[yk]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (x, y)):
            raise ValueError('Coordinates must be finite numbers')
        nx, ny = x / before['original_width'], y / before['original_height']
        points.append({'keys': [xk, yk], 'original_pixels': [x, y],
                       'normalized_xy': [round(nx, 6), round(ny, 6)],
                       'inside_original_image': 0 <= nx < 1 and 0 <= ny < 1})
    return {'images': images, 'current_action_points': points}


def geometry_block(row, data, max_side=512):
    info = json.dumps(geometry(row, data, max_side), ensure_ascii=False, separators=(',', ':'))
    return MARKER + '\n' + INSTRUCTIONS + info + '\n' + END_MARKER + '\n\n'


def augment_row(row, data, max_side=512):
    """Idempotently attach metadata; reject stale or conflicting geometry."""
    result = copy.deepcopy(row)
    text_item = next(x for x in result['messages'][0]['content'] if x['type'] == 'text')
    text = text_item['text']
    block = geometry_block(row, data, max_side)
    if MARKER in text:
        if text.count(MARKER) != 1 or block not in text:
            raise ValueError('Existing geometry differs from the actual images/action')
        return result
    if text.count('=== Context ===') != 1:
        raise ValueError('Expected one original v9 context section')
    text_item['text'] = text.replace('=== Context ===', block + '=== Context ===', 1)
    return result
