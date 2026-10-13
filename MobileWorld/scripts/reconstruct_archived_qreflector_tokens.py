#!/usr/bin/env python3
"""CPU-only retrospective estimates from archived inputs and decoded outputs.

Does not load model weights, regenerate responses, or modify original logs.
"""
from pathlib import Path
import ast
import collections
import csv
import hashlib
import json
import os
import re

os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
                  'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0', 'USE_FLAX': '0',
                  'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
os.nice(10)

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'traj_logs/v2_qverifier_retry'
MODEL = ROOT / '../training/GRPO/qwen3_vl_distilled_dpo_grpo_8b/final'
SOURCE = ROOT / 'src/mobile_world/reflection/base.py'
AUDIT = ROOT / 'artifacts/reports/qreflector_mobileworld_v2_tokens_20261009/audit.json'
OUT = AUDIT.parent / 'reconstructed'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


class SavedAction:
    def __init__(self, action):
        self.action = action

    def model_dump(self, exclude_none=True):
        return {k: v for k, v in self.action.items() if not exclude_none or v is not None}


def prompt_helpers():
    # Read the retained prompt implementation without importing runtime/GPU modules.
    names = {'PROMPT_TEMPLATE', '_MAI_UI_THINKING_RE', '_GENERAL_E2E_THOUGHT_RE'}
    functions = {'extract_subgoal', '_action_to_training_format', 'build_transition_json', 'build_prompt'}
    selected = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in names for n in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    scope = {'json': json, 're': re, 'JSONAction': SavedAction}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), scope)
    assert names | functions <= scope.keys()
    return scope


def main():
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    assert audit['reflector']['calls'] == 3032
    original_hashes = audit['protected_hashes']
    assert all(sha(Path(p)) == h for p, h in original_hashes.items())
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    helpers = prompt_helpers()
    # All 5,991 available ordered screenshots have this size. 73 terminal AFTER
    # screenshots are absent; only their dimensions are inferred, never their pixels.
    dummy = Image.new('RGB', (230, 512))
    image_inputs = processor.image_processor(images=[dummy, dummy], return_tensors='pt')
    grid = image_inputs['image_grid_thw'].tolist()
    image_counts = [int(g.prod()) // processor.image_processor.merge_size**2 for g in image_inputs['image_grid_thw']]
    records = []
    observed_dims = collections.Counter()
    missing_after = []
    sources = dict(original_hashes)
    for file in [SOURCE, Path(__file__), *MODEL.glob('*.json'), MODEL / 'chat_template.jinja']:
        sources[str(file)] = sha(file)
    verified = []
    for path in sorted(LOG.glob('*/traj.json')):
        if '_backup_' in path.parent.name:
            continue
        for episode_id, episode in json.loads(path.read_text()).items():
            if not isinstance(episode, dict):
                continue
            by_step = {row['step']: row for row in episode.get('traj', [])}
            for ref in episode.get('reflections', []):
                step = ref['step']
                row = by_step[step]
                after_missing = False
                for role, image_step in [('before', step), ('after', step + 1)]:
                    image_path = path.parent / 'screenshots' / f'{path.parent.name}-{episode_id}-{image_step}.png'
                    if not image_path.exists():
                        assert role == 'after'
                        after_missing = True
                        missing_after.append({'task': path.parent.name, 'step': step})
                        continue
                    with Image.open(image_path) as picture:
                        observed_dims[str(picture.size)] += 1
                        assert picture.size == (1080, 2400)
                prompt = helpers['build_prompt'](global_task=row['task_goal'],
                    subgoal=helpers['extract_subgoal'](row['prediction']), action=SavedAction(row['action']))
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                    {'type': 'image', 'image': dummy}, {'type': 'image', 'image': dummy}]}]
                serialized = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert serialized.count(processor.image_token) == 2
                expanded = serialized
                for count in image_counts:
                    expanded = expanded.replace(processor.image_token, '<|replay_placeholder|>' * count, 1)
                expanded = expanded.replace('<|replay_placeholder|>', processor.image_token)
                ids = processor.tokenizer(expanded, add_special_tokens=False, truncation=False)['input_ids']
                # Check the fast counting path against complete processor encoding
                # at fixed intervals, without any model or CUDA initialization.
                if len(records) % 300 == 0:
                    encoded = processor.apply_chat_template(messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors='pt', truncation=False)
                    assert ids == encoded['input_ids'][0].tolist()
                    verified.append({'task': path.parent.name, 'step': step, 'tokens': len(ids)})
                raw = ref['raw_output']
                output_ids = processor.tokenizer.encode(raw, add_special_tokens=False)
                records.append({'task': path.parent.name, 'step': step,
                    'reconstructed_input_tokens': len(ids),
                    'retokenized_visible_output_tokens': len(output_ids),
                    'image_tokens': sum(image_counts), 'after_dimensions_inferred': after_missing,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest()})
    assert len(records) == 3032 and len(missing_after) == 73
    assert max(r['reconstructed_input_tokens'] for r in records) < 4096
    total_input = sum(r['reconstructed_input_tokens'] for r in records)
    total_output = sum(r['retokenized_visible_output_tokens'] for r in records)
    actor_input = audit['actor_only']['input_tokens']
    actor_output = audit['actor_only']['output_tokens']
    summary = {
        'status': 'retrospective_estimate_not_original_measured_usage',
        'calls': len(records), 'assigned_tasks': 117,
        'reflector_input_reconstructed': total_input,
        'reflector_output_visible_retokenized': total_output,
        'reflector_mean_input_per_task': total_input / 117,
        'reflector_mean_visible_output_per_task': total_output / 117,
        'actor_input_measured': actor_input, 'actor_output_measured': actor_output,
        'combined_mean_input_per_task_estimated': (actor_input + total_input) / 117,
        'combined_mean_output_per_task_visible_estimated': (actor_output + total_output) / 117,
        'output_one_eos_per_call_scenario': {
            'description': 'Sensitivity scenario only, not a correction or guaranteed upper bound; some calls may stop at token limit.',
            'reflector_mean_output_per_task': (total_output + len(records)) / 117,
            'combined_mean_output_per_task': (actor_output + total_output + len(records)) / 117,
        },
        'image_original_dimensions': [1080, 2400], 'long_side_resize': 512,
        'pil_resized_dimensions': [230, 512], 'image_grid_thw': grid,
        'visual_tokens_per_call': sum(image_counts),
        'observed_screenshot_dimensions': dict(observed_dims),
        'missing_after_images_with_dimensions_inferred': missing_after,
        'full_processor_crosschecks': verified,
        'maximum_input_tokens': max(r['reconstructed_input_tokens'] for r in records),
        'maximum_visible_output_tokens': max(r['retokenized_visible_output_tokens'] for r in records),
        'visible_outputs_at_256_tokens': sum(r['retokenized_visible_output_tokens'] == 256 for r in records),
        'transformers_version_used_for_reconstruction': transformers.__version__,
        'limitations': [
            'The exact serialized historical input IDs, runtime/library version, and prompt fingerprint were not saved. Input counts are conditional on the retained legacy prompt/processor matching the original run.',
            '73 terminal AFTER images are absent. Same simulator resolution as BEFORE is assumed for token counts; pixels are not reconstructed.',
            'Saved outputs were decoded with skip_special_tokens=True and strip(). Re-tokenizing visible text cannot recover original generated IDs, EOS, removed whitespace, or noncanonical tokenization.',
            'Retained checkpoint tokenizer is used; no model inference, network calls, or fresh task outcomes are produced.',
            'Publish these only as estimated/reconstructed usage with a footnote, not as originally measured totals.',
        ],
        'protected_hashes': sources,
    }
    write(OUT / 'summary.json', summary)
    with (OUT / 'by_call.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert all(sha(Path(p)) == h for p, h in sources.items())
    print(json.dumps({k: v for k, v in summary.items() if k not in ['protected_hashes', 'missing_after_images_with_dimensions_inferred']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
