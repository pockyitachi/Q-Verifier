#!/usr/bin/env python3
"""Reconstruct GUI-Reflection usage on CPU from the archived 117-task run."""
from pathlib import Path
import ast
import collections
import csv
import hashlib
import json
import math
import os
import re
import sys

from reconstruct_archived_qreflector_tokens import SavedAction, prompt_helpers, SOURCE

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/evaluations/gui_reflection/mobileworld_gui117_20260930_1705'
MODEL = ROOT / 'artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain'
OFFICIAL = ROOT / 'artifacts/vendor/GUI_Reflection/internvl_chat'
PRIOR = ROOT / 'artifacts/reports/qreflector_mobileworld_table_audit_20261007/audit.json'
OUT = ROOT / 'artifacts/reports/gui_reflection_mobileworld_tokens_reconstructed_20261009'
os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0', 'USE_FLAX': '0',
    'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'HF_HOME': str(OUT / 'cache'), 'HF_MODULES_CACHE': str(OUT / 'cache/modules'),
    'HUGGINGFACE_HUB_CACHE': str(OUT / 'cache/hub')})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def extract_class(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Transformers may create its local cache during import.
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'summary.json').exists() and not (OUT / 'by_call.csv').exists()
    config = read(RUN / 'config.json')
    audit = read(PRIOR)
    assert config['reflector_prompt'] == 'shared'
    assert sha(RUN / 'summary.json') == audit['source_summary_sha256']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True, local_files_only=True)
    sys.path.insert(0, str(OFFICIAL))
    from internvl.conversation import get_conv_template
    preprocess_file = OFFICIAL / 'eval/preprocess.py'
    infer_file = OFFICIAL / 'eval/infer.py'
    Preprocessor = extract_class(preprocess_file, 'ImageProcessor', {'math': math})
    OfficialModel = extract_class(infer_file, 'Model', {'torch': torch, 're': re, 'get_conv_template': get_conv_template})
    preprocessor = Preprocessor()
    model_config = read(MODEL / 'config.json')
    image_size = model_config['force_image_size'] or model_config['vision_config']['image_size']
    per_tile = int((image_size // model_config['vision_config']['patch_size']) ** 2 * model_config['downsample_ratio'] ** 2)
    dummy = Image.new('RGB', (230, 512))
    tile_count = len(preprocessor.dynamic_preprocess_v3(dummy, image_size=image_size, max_num=6, use_thumbnail=True))
    image_block = '<img>' + '<IMG_CONTEXT>' * per_tile * tile_count + '<|im_end|>'
    helpers = prompt_helpers()

    class CaptureOnly:
        def generate(self, **kwargs):
            self.input_ids = kwargs['input_ids'][0].tolist()
            self.tile_count = kwargs['pixel_values'].shape[0]
            return torch.tensor([tokenizer.encode('Yes', add_special_tokens=False) + [tokenizer.eos_token_id]])

    # This instance has no weights. Its generate stub only captures input IDs.
    harness = OfficialModel.__new__(OfficialModel)
    harness.tokenizer = tokenizer
    harness.preprocessor = preprocessor
    harness.model = CaptureOnly()
    harness.image_size = image_size
    harness.use_thumbnail = True
    harness.max_num = 6
    harness.num_image_token = per_tile
    harness.template_name = 'internlm2-chat-v3'
    harness.transform = lambda image: torch.zeros(3, image_size, image_size)

    files = [PRIOR, RUN / 'config.json', RUN / 'summary.json', SOURCE, Path(__file__),
        ROOT / 'scripts/reconstruct_archived_qreflector_tokens.py', infer_file, preprocess_file,
        OFFICIAL / 'internvl/conversation.py', *MODEL.glob('*.json'), *MODEL.glob('tokenization*.py')]
    protected = {str(p): sha(p) for p in files}
    counts = []
    dims = collections.Counter()
    outputs = collections.Counter()
    missing_after = []
    checks = []
    actor_input = actor_output = actor_calls = 0
    by_task = []
    assert not list((RUN / 'trajectories').glob('*_backup_*'))
    for path in sorted((RUN / 'trajectories').glob('*/traj.json')):
        protected[str(path)] = sha(path)
        task_input = task_output = 0
        for episode_id, episode in read(path).items():
            if not isinstance(episode, dict):
                continue
            usage = episode.get('token_usage') or {}
            task_input += usage.get('prompt_tokens', 0)
            task_output += usage.get('completion_tokens', 0)
            steps = {r['step']: r for r in episode.get('traj', [])}
            for reflection in episode.get('reflections', []):
                step = reflection['step']
                row = steps[step]
                inferred_after = False
                for role, index in [('before', step), ('after', step + 1)]:
                    path_image = path.parent / 'screenshots' / f'{path.parent.name}-{episode_id}-{index}.png'
                    if not path_image.exists():
                        assert role == 'after'
                        inferred_after = True
                        missing_after.append({'task': path.parent.name, 'step': step})
                        continue
                    with Image.open(path_image) as picture:
                        dims[str(picture.size)] += 1
                        assert picture.size == (1080, 2400)
                prompt = helpers['build_prompt'](global_task=row['task_goal'],
                    subgoal=helpers['extract_subgoal'](row['prediction']), action=SavedAction(row['action']))
                assert '<image>' not in prompt
                question = image_block + '\n' + image_block + '\n' + prompt
                conv = get_conv_template('internlm2-chat-v3')
                conv.system_message = ''
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()
                ids = tokenizer(query, add_special_tokens=False)['input_ids']
                assert ids.count(tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')) == per_tile * tile_count * 2
                if len(counts) % 400 == 0:
                    original_cuda = torch.Tensor.cuda
                    try:
                        # Process-local CPU identity; no CUDA calls or model inference.
                        torch.Tensor.cuda = lambda self, *args, **kwargs: self
                        response = harness(prompt, [dummy, dummy], {'max_new_tokens': 256, 'do_sample': False})
                    finally:
                        torch.Tensor.cuda = original_cuda
                    assert response == 'Yes'
                    assert harness.model.input_ids == ids
                    assert harness.model.tile_count == 2 * tile_count
                    checks.append({'task': path.parent.name, 'step': step, 'input_tokens': len(ids)})
                raw = reflection['raw_output']
                outputs[raw] += 1
                count = len(tokenizer.encode(raw, add_special_tokens=False))
                counts.append({'task': path.parent.name, 'step': step, 'reconstructed_input_tokens': len(ids),
                    'retokenized_visible_output_tokens': count, 'raw_output': raw,
                    'visual_tokens': per_tile * tile_count * 2, 'after_dimensions_inferred': inferred_after,
                    'prompt_sha256': hashlib.sha256(query.encode()).hexdigest()})
        log_input = log_output = 0
        for log in path.parent.glob('*.log'):
            protected[str(log)] = sha(log)
            calls = re.findall(r'OpenAI API usage: completion=(\d+), prompt=(\d+)', log.read_text(errors='replace'))
            actor_calls += len(calls)
            log_input += sum(int(c[1]) for c in calls)
            log_output += sum(int(c[0]) for c in calls)
        assert (log_input, log_output) == (task_input, task_output), path
        actor_input += task_input
        actor_output += task_output
        rs = [r for r in counts if r['task'] == path.parent.name]
        by_task.append({'task': path.parent.name, 'actor_input_measured': task_input,
            'actor_output_measured': task_output, 'reflector_calls': len(rs),
            'reflector_input_reconstructed': sum(r['reconstructed_input_tokens'] for r in rs),
            'reflector_output_visible_retokenized': sum(r['retokenized_visible_output_tokens'] for r in rs)})
    assert len(by_task) == 117 and len(counts) == 3836 and len(missing_after) == 73
    assert (actor_input, actor_output, actor_calls) == (37050847, 230564, 3938)
    assert outputs == {'Yes': 3400, 'No': 436}
    total_input = sum(r['reconstructed_input_tokens'] for r in counts)
    total_output = sum(r['retokenized_visible_output_tokens'] for r in counts)
    summary = {
        'status': 'retrospective_estimate_not_original_measured_usage',
        'host': 'Tongyi-MAI/MAI-UI-8B', 'reflector': 'craigwu/GUI_Reflection_8b_pretrain',
        'prompt_mode': 'shared', 'assigned_tasks': 117, 'successes': 29, 'success_pct': 29 / 117 * 100,
        'reflector_calls': len(counts), 'output_counts': dict(outputs),
        'actor_input_measured': actor_input, 'actor_output_measured': actor_output,
        'actor_calls': actor_calls, 'actor_log_trajectory_totals_match': True,
        'reflector_input_reconstructed': total_input, 'reflector_output_visible_retokenized': total_output,
        'actor_mean_input_per_task': actor_input / 117, 'actor_mean_output_per_task': actor_output / 117,
        'reflector_mean_input_per_task': total_input / 117,
        'reflector_mean_visible_output_per_task': total_output / 117,
        'combined_mean_input_per_task_estimated': (actor_input + total_input) / 117,
        'combined_mean_output_per_task_visible_estimated': (actor_output + total_output) / 117,
        'output_one_eos_per_call_scenario': {
            'description': 'Sensitivity scenario only, not recovered original generated token IDs.',
            'reflector_mean_output_per_task': (total_output + len(counts)) / 117,
            'combined_mean_output_per_task': (actor_output + total_output + len(counts)) / 117},
        'image_original_dimensions': [1080, 2400], 'image_resized_dimensions': [230, 512],
        'image_size': image_size, 'tiles_per_image': tile_count, 'visual_tokens_per_tile': per_tile,
        'visual_tokens_per_call': per_tile * tile_count * 2,
        'observed_screenshot_dimensions': dict(dims),
        'missing_after_images_with_dimensions_inferred': missing_after,
        'official_wrapper_capture_crosschecks': checks,
        'transformers_version': transformers.__version__,
        'tokenizer_class': type(tokenizer).__name__,
        'historical_prompt_source_file_hash': config['source_sha256']['src/mobile_world/reflection/base.py'],
        'current_prompt_source_file_hash': sha(SOURCE),
        'limitations': [
            'Prompt source file hash differs from the archived config. The retained shared prompt/helper implementation is used; exact historical serialized inputs were not saved.',
            '73 terminal AFTER screenshots are absent; their dimensions are inferred to match the fixed emulator resolution.',
            'Visible Yes/No text is re-tokenized. Historical EOS/special IDs and stripped whitespace cannot be recovered; body-only output estimates exclude them.',
            'CPU capture crosschecks execute the official serialization with a stub model. They validate reconstruction code, not the historical hidden generation sequence.',
            'Use as estimated/reconstructed usage with a footnote, not as original measured token totals.',
        ], 'protected_hashes': protected,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for name, rs in [('by_call.csv', counts), ('by_task.csv', by_task)]:
        with (OUT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rs[0]))
            writer.writeheader()
            writer.writerows(rs)
    assert all(sha(Path(p)) == h for p, h in protected.items())
    print(json.dumps({k: v for k, v in summary.items() if k not in ['protected_hashes', 'missing_after_images_with_dimensions_inferred', 'official_wrapper_capture_crosschecks']}, indent=2))


if __name__ == '__main__':
    main()
