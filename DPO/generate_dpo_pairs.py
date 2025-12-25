#!/usr/bin/env python3
"""
Step 3+4: Generate DPO Preference Pairs via Data Corruption

This script:
1. Takes YES samples from training data
2. Creates corrupted versions (swap AFTER images)
3. Generates preference pairs for DPO training

Corruption types:
- Cross-task swap: AFTER image from completely different task
- Same-task wrong-step: AFTER from different step in same task
- No-change: AFTER = BEFORE (action had no effect)

Output format for DPO:
{
    "prompt": "...[BEFORE image + subgoal + action]...",
    "chosen": "Yes, valid transition because...",
    "rejected": "No, invalid transition because...",
    "chosen_images": [...],
    "rejected_images": [...]
}
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import hashlib


def load_training_data(data_path: str) -> List[dict]:
    """Load training samples from JSONL."""
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_sample_info(sample: dict) -> dict:
    """Extract key info from a sample."""
    content = sample['messages'][0]['content']
    
    # Get text and images
    text = ""
    images = []
    for item in content:
        if item['type'] == 'text':
            text = item['text']
        elif item['type'] == 'image':
            images.append(item['image'])
    
    # Extract task
    task_start = text.find('=== Global Task ===')
    task_end = text.find('===', task_start + 20) if task_start != -1 else -1
    task = text[task_start+20:task_end].strip() if task_start != -1 else ""
    
    # Extract subgoal
    subgoal_start = text.find('=== Current Subgoal ===')
    subgoal_end = text.find('===', subgoal_start + 24) if subgoal_start != -1 else -1
    subgoal = text[subgoal_start+24:subgoal_end].strip() if subgoal_start != -1 else ""
    
    # Get answer
    answer_text = sample['messages'][1]['content'][0]['text']
    try:
        answer_data = json.loads(answer_text)
        answer = answer_data.get('answer', '').lower()
        reason = answer_data.get('reason', '')
    except:
        answer = 'yes' if 'yes' in answer_text.lower() else 'no'
        reason = answer_text
    
    return {
        'text': text,
        'images': images,  # [before, after]
        'task': task,
        'subgoal': subgoal,
        'answer': answer,
        'reason': reason,
        'full_sample': sample
    }


def get_task_hash(task: str) -> str:
    """Create a short hash for task grouping."""
    return hashlib.md5(task.encode()).hexdigest()[:8]


def generate_corruption_reason(corruption_type: str, original_subgoal: str, 
                               source_task: str = None) -> str:
    """Generate a reason for why the corrupted sample is invalid."""
    
    if corruption_type == "cross_task":
        return (f"The AFTER screenshot shows a completely different context that is unrelated "
                f"to the subgoal '{original_subgoal[:50]}...'. The action did not make progress "
                f"toward the intended goal.")
    
    elif corruption_type == "wrong_step":
        return (f"The AFTER screenshot shows a state that doesn't match the expected outcome "
                f"of the action. The screen changed to an unexpected state that doesn't align "
                f"with the subgoal '{original_subgoal[:50]}...'.")
    
    elif corruption_type == "no_change":
        return (f"The BEFORE and AFTER screenshots are identical, indicating the action had "
                f"no effect. The subgoal '{original_subgoal[:50]}...' requires a visible change "
                f"but none occurred.")
    
    return "The transition is invalid."


def create_dpo_pair(positive_sample: dict, corrupted_after: str, 
                    corruption_type: str, positive_images: List[str]) -> dict:
    """Create a DPO preference pair."""
    
    info = extract_sample_info(positive_sample)
    before_image = info['images'][0]  # BEFORE image stays the same
    original_after = info['images'][1]  # Original correct AFTER
    
    # Build the prompt (shared between chosen and rejected)
    # We'll include the BEFORE image and text, but AFTER differs
    text = info['text']
    
    # Find where AFTER description starts and truncate for prompt
    # The prompt should be everything up to the point of judgment
    prompt_text = text  # Full context
    
    # Chosen response (original correct)
    chosen_response = json.dumps({
        "answer": "Yes",
        "reason": info['reason']
    })
    
    # Rejected response (corrupted)
    rejected_reason = generate_corruption_reason(
        corruption_type, 
        info['subgoal']
    )
    rejected_response = json.dumps({
        "answer": "No",
        "reason": rejected_reason
    })
    
    return {
        "prompt": prompt_text,
        "prompt_images": [before_image],  # Only BEFORE in prompt
        "chosen": chosen_response,
        "chosen_after_image": original_after,  # Correct AFTER
        "rejected": rejected_response,
        "rejected_after_image": corrupted_after,  # Corrupted AFTER
        "corruption_type": corruption_type,
        "original_task": info['task'][:50],
        "subgoal": info['subgoal'][:80]
    }


def generate_pairs(samples: List[dict], 
                   corruptions_per_sample: int = 2,
                   seed: int = 42) -> List[dict]:
    """Generate DPO pairs from positive samples."""
    
    random.seed(seed)
    
    # Extract info from all samples
    sample_infos = [extract_sample_info(s) for s in samples]
    
    # Filter to YES samples only
    positive_samples = [(s, info) for s, info in zip(samples, sample_infos) 
                        if info['answer'] == 'yes']
    
    print(f"Found {len(positive_samples)} YES samples for pair generation")
    
    # Group by task for same-task corruption
    task_groups = defaultdict(list)
    for s, info in positive_samples:
        task_hash = get_task_hash(info['task'])
        task_groups[task_hash].append((s, info))
    
    # Collect all AFTER images for cross-task corruption
    all_after_images = [(info['images'][1], info['task']) 
                        for _, info in positive_samples]
    
    pairs = []
    
    for sample, info in positive_samples:
        sample_pairs = []
        before_image = info['images'][0]
        original_after = info['images'][1]
        task_hash = get_task_hash(info['task'])
        
        # Corruption 1: Cross-task swap
        # Find AFTER image from a different task
        other_task_afters = [(img, task) for img, task in all_after_images 
                             if get_task_hash(task) != task_hash]
        
        if other_task_afters:
            corrupted_after, source_task = random.choice(other_task_afters)
            pair = create_dpo_pair(sample, corrupted_after, "cross_task", info['images'])
            pair['source_task'] = source_task[:50]
            sample_pairs.append(pair)
        
        # Corruption 2: Same-task wrong-step (if enough samples in same task)
        same_task_samples = task_groups[task_hash]
        other_steps = [(s, i) for s, i in same_task_samples 
                       if i['images'][1] != original_after]
        
        if other_steps and len(sample_pairs) < corruptions_per_sample:
            _, other_info = random.choice(other_steps)
            corrupted_after = other_info['images'][1]
            pair = create_dpo_pair(sample, corrupted_after, "wrong_step", info['images'])
            sample_pairs.append(pair)
        
        # Corruption 3: No-change (AFTER = BEFORE)
        if len(sample_pairs) < corruptions_per_sample:
            pair = create_dpo_pair(sample, before_image, "no_change", info['images'])
            sample_pairs.append(pair)
        
        # Add up to corruptions_per_sample pairs
        pairs.extend(sample_pairs[:corruptions_per_sample])
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument("--data_path", type=str, 
                        default="../SFT/training_data_filtered/train.jsonl",
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./dpo_pairs",
                        help="Output directory")
    parser.add_argument("--corruptions_per_sample", type=int, default=2,
                        help="Number of corruptions per positive sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_gpt_reasoning", action="store_true",
                        help="Use GPT to generate rejection reasoning (requires API)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DPO PAIR GENERATION")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Corruptions per sample: {args.corruptions_per_sample}")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading training data...")
    samples = load_training_data(args.data_path)
    print(f"Loaded {len(samples)} samples")
    
    # Generate pairs
    print("\n[2/3] Generating DPO pairs...")
    pairs = generate_pairs(
        samples, 
        corruptions_per_sample=args.corruptions_per_sample,
        seed=args.seed
    )
    print(f"Generated {len(pairs)} preference pairs")
    
    # Analyze corruption types
    type_counts = defaultdict(int)
    for p in pairs:
        type_counts[p['corruption_type']] += 1
    
    print("\nCorruption type distribution:")
    for ctype, count in type_counts.items():
        print(f"  {ctype}: {count} ({100*count/len(pairs):.1f}%)")
    
    # Save pairs
    print("\n[3/3] Saving pairs...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, "dpo_pairs.jsonl")
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"\n✅ Saved {len(pairs)} pairs to {output_path}")
    
    # Save summary
    summary = {
        "total_pairs": len(pairs),
        "source_samples": len(samples),
        "corruptions_per_sample": args.corruptions_per_sample,
        "corruption_types": dict(type_counts),
        "seed": args.seed
    }
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved summary to {summary_path}")
    
    # Show sample pair
    print("\n" + "=" * 60)
    print("SAMPLE PAIR")
    print("=" * 60)
    sample_pair = pairs[0]
    print(f"Corruption type: {sample_pair['corruption_type']}")
    print(f"Subgoal: {sample_pair['subgoal']}")
    print(f"Chosen (correct): {sample_pair['chosen'][:100]}...")
    print(f"Rejected (corrupted): {sample_pair['rejected'][:100]}...")


if __name__ == "__main__":
    main()

