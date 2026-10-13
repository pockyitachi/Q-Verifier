#!/usr/bin/env python3
"""
Generate Corrupted Negative Instances via Trajectory Corruption

Strategy:
- Trajectory: A → B → C → D → E (correct path, all positive)
- Each transition is a positive instance (labeled "Yes")
- Corrupt at step N: Replace AFTER image with wrong image
- This creates a "wrong branch" - all subsequent steps become invalid

Corruption Types:
1. Cross-task: AFTER from completely different task
2. Wrong-step (future): AFTER from a LATER step in same task (skipped ahead)
3. Wrong-step (past): AFTER from an EARLIER step in same task (went back)
4. No-change: AFTER = BEFORE (action had no effect)

Output: Balanced dataset with ~50% positive (Yes) and ~50% negative (No)
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import hashlib
from copy import deepcopy


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


def generate_rejection_reason(corruption_type: str, original_subgoal: str) -> str:
    """Generate a reason for why the corrupted sample is invalid."""
    
    reasons = {
        "cross_task": [
            f"The AFTER screenshot shows a completely different application context that is unrelated to the subgoal '{original_subgoal[:50]}...'. The action did not produce the expected result.",
            f"The screen transition shows content from a different task entirely. This does not align with the goal of '{original_subgoal[:50]}...'.",
            f"The AFTER state belongs to a different workflow. The expected outcome for '{original_subgoal[:50]}...' was not achieved.",
        ],
        "wrong_step_future": [
            f"The AFTER screenshot shows a state that appears to skip ahead in the workflow. The immediate result of the action does not match the subgoal '{original_subgoal[:50]}...'.",
            f"The transition jumped to a later state in the task flow. This is not the direct outcome of the described action.",
            f"The screen shows a future state that requires intermediate steps. The action did not directly produce this result.",
        ],
        "wrong_step_past": [
            f"The AFTER screenshot shows a previous state in the workflow. The action appears to have gone backwards instead of progressing toward '{original_subgoal[:50]}...'.",
            f"The transition reverted to an earlier screen state. This does not represent progress on the current subgoal.",
            f"The screen shows a state that was already passed. The expected forward progression did not occur.",
        ],
        "no_change": [
            f"The BEFORE and AFTER screenshots are identical. The action had no visible effect, but the subgoal '{original_subgoal[:50]}...' requires a change.",
            f"No change occurred between states. The action failed to execute or had no impact on the screen.",
            f"The screen remained unchanged after the action. This indicates the action did not successfully complete.",
        ],
    }
    
    return random.choice(reasons.get(corruption_type, ["The transition is invalid."]))


def create_corrupted_sample(original_sample: dict, corrupted_after: str, 
                            corruption_type: str, original_info: dict) -> dict:
    """Create a corrupted negative sample."""
    
    # Deep copy the original sample
    corrupted = deepcopy(original_sample)
    
    # Replace the AFTER image (second image in content)
    content = corrupted['messages'][0]['content']
    image_count = 0
    for i, item in enumerate(content):
        if item['type'] == 'image':
            image_count += 1
            if image_count == 2:  # Second image is AFTER
                content[i]['image'] = corrupted_after
                break
    
    # Update the answer to "No" with rejection reason
    rejection_reason = generate_rejection_reason(corruption_type, original_info['subgoal'])
    corrupted['messages'][1]['content'][0]['text'] = json.dumps({
        "answer": "No",
        "reason": rejection_reason
    })
    
    return corrupted


def generate_corruptions(samples: List[dict], 
                         target_negative_ratio: float = 0.5,
                         seed: int = 42) -> Tuple[List[dict], dict]:
    """
    Generate corrupted negative samples to balance the dataset.
    
    Args:
        samples: Original training samples
        target_negative_ratio: Target ratio of negative samples (0.5 = 50%)
        seed: Random seed
    
    Returns:
        Tuple of (all_samples, stats)
    """
    random.seed(seed)
    
    # Extract info from all samples
    sample_infos = [(s, extract_sample_info(s)) for s in samples]
    
    # Count existing positives and negatives
    positive_samples = [(s, info) for s, info in sample_infos if info['answer'] == 'yes']
    negative_samples = [(s, info) for s, info in sample_infos if info['answer'] == 'no']
    
    print(f"Original distribution:")
    print(f"  Positive (Yes): {len(positive_samples)}")
    print(f"  Negative (No): {len(negative_samples)}")
    
    # Calculate how many negatives we need
    total_after = len(samples)
    current_neg_ratio = len(negative_samples) / total_after
    
    if current_neg_ratio >= target_negative_ratio:
        print(f"Already have {current_neg_ratio:.1%} negatives, no corruption needed")
        return samples, {"corruptions_added": 0}
    
    # Calculate needed negatives
    # After adding N corruptions: neg_ratio = (existing_neg + N) / (total + N)
    # Solve for N: N = (target * total - existing_neg) / (1 - target)
    needed = int((target_negative_ratio * total_after - len(negative_samples)) / (1 - target_negative_ratio))
    needed = max(0, needed)
    
    print(f"\nNeed to generate {needed} negative samples to reach {target_negative_ratio:.0%} ratio")
    
    # Group samples by task for trajectory-aware corruption
    task_groups = defaultdict(list)
    for s, info in positive_samples:
        task_hash = get_task_hash(info['task'])
        task_groups[task_hash].append((s, info))
    
    # Collect all AFTER images by task
    all_after_images = {}
    for task_hash, samples_in_task in task_groups.items():
        all_after_images[task_hash] = [info['images'][1] for _, info in samples_in_task]
    
    # Get all AFTER images from all tasks for cross-task corruption
    all_afters_flat = [(info['images'][1], get_task_hash(info['task'])) 
                       for _, info in positive_samples]
    
    # Generate corruptions
    corrupted_samples = []
    corruption_stats = defaultdict(int)
    
    # Shuffle positive samples for random selection
    shuffled_positives = list(positive_samples)
    random.shuffle(shuffled_positives)
    
    corruption_idx = 0
    while len(corrupted_samples) < needed and corruption_idx < len(shuffled_positives) * 4:
        # Cycle through positive samples
        sample, info = shuffled_positives[corruption_idx % len(shuffled_positives)]
        
        task_hash = get_task_hash(info['task'])
        before_image = info['images'][0]
        original_after = info['images'][1]
        
        # Choose corruption type
        corruption_type = random.choice(["cross_task", "wrong_step_future", "wrong_step_past", "no_change"])
        
        corrupted_after = None
        
        if corruption_type == "cross_task":
            # AFTER from a different task
            other_afters = [(img, t) for img, t in all_afters_flat if t != task_hash]
            if other_afters:
                corrupted_after, _ = random.choice(other_afters)
        
        elif corruption_type in ["wrong_step_future", "wrong_step_past"]:
            # AFTER from different step in same task
            same_task_afters = [img for img in all_after_images.get(task_hash, []) 
                               if img != original_after]
            if same_task_afters:
                corrupted_after = random.choice(same_task_afters)
        
        elif corruption_type == "no_change":
            # AFTER = BEFORE
            corrupted_after = before_image
        
        if corrupted_after and corrupted_after != original_after:
            corrupted_sample = create_corrupted_sample(sample, corrupted_after, corruption_type, info)
            corrupted_samples.append(corrupted_sample)
            corruption_stats[corruption_type] += 1
        
        corruption_idx += 1
    
    # Combine original and corrupted
    all_samples = samples + corrupted_samples
    
    # Shuffle final dataset
    random.shuffle(all_samples)
    
    final_stats = {
        "original_total": len(samples),
        "original_positive": len(positive_samples),
        "original_negative": len(negative_samples),
        "corruptions_added": len(corrupted_samples),
        "corruption_types": dict(corruption_stats),
        "final_total": len(all_samples),
        "final_negative_ratio": (len(negative_samples) + len(corrupted_samples)) / len(all_samples)
    }
    
    return all_samples, final_stats


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted negative instances")
    parser.add_argument("--data_path", type=str, 
                        default="../SFT/training_data_filtered/train.jsonl",
                        help="Path to original training data")
    parser.add_argument("--output_dir", type=str, default="./corrupted_data",
                        help="Output directory")
    parser.add_argument("--target_ratio", type=float, default=0.5,
                        help="Target negative sample ratio (0.5 = balanced)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA CORRUPTION: GENERATING NEGATIVE INSTANCES")
    print("=" * 60)
    print(f"Input: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Target negative ratio: {args.target_ratio:.0%}")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading training data...")
    samples = load_training_data(args.data_path)
    print(f"Loaded {len(samples)} samples")
    
    # Generate corruptions
    print("\n[2/3] Generating corrupted samples...")
    all_samples, stats = generate_corruptions(
        samples, 
        target_negative_ratio=args.target_ratio,
        seed=args.seed
    )
    
    # Save
    print("\n[3/3] Saving balanced dataset...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, "train_balanced.jsonl")
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Saved {len(all_samples)} samples to {output_path}")
    
    # Save stats
    stats_path = os.path.join(args.output_dir, "corruption_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved stats to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original samples: {stats['original_total']}")
    print(f"  - Positive: {stats['original_positive']}")
    print(f"  - Negative: {stats['original_negative']}")
    print(f"Corruptions added: {stats['corruptions_added']}")
    print(f"  By type:")
    for ctype, count in stats['corruption_types'].items():
        print(f"    - {ctype}: {count}")
    print(f"Final dataset: {stats['final_total']} samples")
    print(f"Final negative ratio: {stats['final_negative_ratio']:.1%}")


if __name__ == "__main__":
    main()

