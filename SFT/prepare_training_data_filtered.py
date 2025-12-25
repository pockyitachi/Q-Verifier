#!/usr/bin/env python3
"""
Prepare trajectory data for Qwen3-VL verifier training.
FILTERED VERSION: Only includes previous steps with the SAME subgoal.

This script:
1. Collects all trajectory data from trajectory_data/
2. For each step, finds ALL previous steps with the same subgoal
3. Rebuilds messages with matching previous steps
4. Copies images and outputs train.jsonl and val.jsonl

Usage:
    python prepare_training_data_filtered.py --output_dir training_data_filtered --val_split 0.1
"""

import json
import os
import shutil
import argparse
import random
import re
from pathlib import Path


def extract_subgoal(message_text: str) -> str:
    """Extract the Current Subgoal from message text."""
    match = re.search(r'=== Current Subgoal ===\n(.+?)\n\n===', message_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def build_message_text(global_task: str, current_subgoal: str, transition: dict, prev_steps: list) -> str:
    """Build the full message text for a training sample."""
    
    # Build current transition (without step number, before/after image, outcome, reason)
    current_transition = {
        "from_state": transition.get("from_state", {}),
        "action": transition.get("action", {}),
        "to_state": transition.get("to_state", {})
    }
    
    # Remove 'valid' from to_state if present (we keep it in prev_steps)
    if "valid" in current_transition["to_state"]:
        current_transition["to_state"] = {k: v for k, v in current_transition["to_state"].items() if k != "valid"}
    
    # Build prev_steps for display (include 'valid' in to_state)
    prev_steps_display = []
    for ps in prev_steps:
        prev_step_data = {
            "step": ps.get("step"),
            "from_state": ps.get("from_state", {}),
            "action": ps.get("action", {}),
            "to_state": ps.get("to_state", {})
        }
        prev_steps_display.append(prev_step_data)
    
    text = f"""You are a mobile GUI reflector.
Your job is to determine whether the current state transition is valid — meaning whether it aligns with both the user's global task and the current subgoal.

=== Global Task ===
{global_task}

=== Current Subgoal ===
{current_subgoal}

=== Current Transition ===
{json.dumps(current_transition, indent=2)}

=== Previous Steps (latest {len(prev_steps_display)}) ===
{json.dumps(prev_steps_display, indent=2)}

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Question ===
Is this transition valid and aligned with the current subgoal?
Answer with "Yes" or "No" and provide a brief reason explaining why."""
    
    return text


def prepare_training_data_filtered(
    trajectory_dir: str = "trajectory_data",
    output_dir: str = "training_data_filtered",
    val_split: float = 0.1,
    seed: int = 42,
    max_prev_steps: int = 5
):
    """
    Prepare training data with filtered previous steps (same subgoal only).
    """
    random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PREPARING TRAINING DATA (FILTERED BY SUBGOAL)")
    print("=" * 60)
    print(f"Source: {trajectory_dir}")
    print(f"Output: {output_dir}")
    print(f"Val split: {val_split * 100:.0f}%")
    print(f"Max prev steps: {max_prev_steps}")
    print()
    
    all_samples = []
    total_images_copied = 0
    tasks_processed = 0
    
    stats = {
        "total_prev_steps_added": 0,
        "max_prev_steps": 0,
        "samples_with_prev_steps": 0
    }
    
    # Process each task folder
    trajectory_path = Path(trajectory_dir)
    for task_dir in sorted(trajectory_path.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_id = task_dir.name
        
        # Find trajectory JSON file
        json_files = list(task_dir.glob("trajectory_*.json"))
        if not json_files:
            print(f"⚠️  Skipping {task_id}: No JSON file")
            continue
        
        json_path = json_files[0]
        
        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping {task_id}: Invalid JSON - {e}")
            continue
        
        messages = data.get("messages", [])
        transitions = data.get("transitions", [])
        global_task = data.get("instruction", "")
        
        if not messages or not transitions:
            print(f"⚠️  Skipping {task_id}: No messages or transitions")
            continue
        
        # Build step -> subgoal mapping from original messages
        step_subgoals = {}
        for i, sample in enumerate(messages):
            step_num = i + 1
            if "messages" in sample and len(sample["messages"]) > 0:
                user_content = sample["messages"][0].get("content", [])
                for item in user_content:
                    if item.get("type") == "text":
                        subgoal = extract_subgoal(item["text"])
                        step_subgoals[step_num] = subgoal
                        break
        
        # Build step -> transition mapping
        step_transitions = {t["step"]: t for t in transitions}
        
        # Process each sample
        for i, sample in enumerate(messages):
            step_num = i + 1
            current_subgoal = step_subgoals.get(step_num, "")
            current_transition = step_transitions.get(step_num, {})
            
            if not current_subgoal or not current_transition:
                continue
            
            # Find ALL previous steps with the SAME subgoal
            matching_prev_steps = []
            for prev_step_num in range(1, step_num):  # All steps before current
                prev_subgoal = step_subgoals.get(prev_step_num, "")
                if prev_subgoal == current_subgoal:
                    prev_transition = step_transitions.get(prev_step_num, {})
                    if prev_transition:
                        matching_prev_steps.append(prev_transition)
            
            # Limit to max_prev_steps (keep the most recent ones)
            if len(matching_prev_steps) > max_prev_steps:
                matching_prev_steps = matching_prev_steps[-max_prev_steps:]
            
            # Stats
            if matching_prev_steps:
                stats["samples_with_prev_steps"] += 1
                stats["total_prev_steps_added"] += len(matching_prev_steps)
                stats["max_prev_steps"] = max(stats["max_prev_steps"], len(matching_prev_steps))
            
            # Build the new message text
            new_text = build_message_text(
                global_task=global_task,
                current_subgoal=current_subgoal,
                transition=current_transition,
                prev_steps=matching_prev_steps
            )
            
            # Get the original answer
            original_answer = None
            if "messages" in sample and len(sample["messages"]) > 1:
                assistant_content = sample["messages"][1].get("content", [])
                for item in assistant_content:
                    if item.get("type") == "text":
                        original_answer = item["text"]
                        break
            
            if not original_answer:
                continue
            
            # Build image paths
            before_image = current_transition.get("before_image", "")
            after_image = current_transition.get("after_image", "")
            
            # Copy images and update paths
            new_before_path = ""
            new_after_path = ""
            
            if before_image:
                before_filename = os.path.basename(before_image)
                new_before_filename = f"{task_id}_{before_filename}"
                new_before_path = f"images/{new_before_filename}"
                
                src = task_dir / before_image
                dst = images_path / new_before_filename
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    total_images_copied += 1
            
            if after_image:
                after_filename = os.path.basename(after_image)
                new_after_filename = f"{task_id}_{after_filename}"
                new_after_path = f"images/{new_after_filename}"
                
                src = task_dir / after_image
                dst = images_path / new_after_filename
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    total_images_copied += 1
            
            # Build new sample
            new_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": new_text},
                            {"type": "image", "image": new_before_path},
                            {"type": "image", "image": new_after_path}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": original_answer}
                        ]
                    }
                ]
            }
            
            all_samples.append(new_sample)
        
        tasks_processed += 1
        print(f"✓ {task_id}: {len(messages)} samples")
    
    print()
    print(f"Processed {tasks_processed} tasks")
    print(f"Total samples: {len(all_samples)}")
    print(f"Images copied: {total_images_copied}")
    
    print()
    print("Previous Steps Stats (same-subgoal filtering):")
    print(f"  Samples with previous steps: {stats['samples_with_prev_steps']}")
    print(f"  Total previous steps added: {stats['total_prev_steps_added']}")
    print(f"  Max previous steps in one sample: {stats['max_prev_steps']}")
    if stats['samples_with_prev_steps'] > 0:
        avg = stats['total_prev_steps_added'] / stats['samples_with_prev_steps']
        print(f"  Avg previous steps (when present): {avg:.1f}")
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split into train/val
    val_size = int(len(all_samples) * val_split)
    train_size = len(all_samples) - val_size
    
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    
    print()
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Write JSONL files
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"
    
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    if val_samples:
        with open(val_path, "w") as f:
            for sample in val_samples:
                f.write(json.dumps(sample) + "\n")
    
    # Write metadata
    metadata = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "total_images": total_images_copied,
        "tasks_processed": tasks_processed,
        "val_split": val_split,
        "seed": seed,
        "filtering": {
            "method": "same_subgoal_only",
            "description": f"Previous steps only include steps with the same subgoal (max {max_prev_steps})",
            "max_prev_steps_limit": max_prev_steps,
            "samples_with_prev_steps": stats["samples_with_prev_steps"],
            "total_prev_steps_added": stats["total_prev_steps_added"],
            "max_prev_steps_actual": stats["max_prev_steps"]
        }
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"✓ {train_path}")
    if val_samples:
        print(f"✓ {val_path}")
    print(f"✓ {metadata_path}")
    print(f"✓ {images_path}/ ({total_images_copied} images)")
    print()
    print("Done! Ready for Qwen3-VL training (filtered version).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data (filtered by subgoal)")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectory_data",
        help="Path to trajectory data folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_data_filtered",
        help="Path to output folder"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (0.0 to 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_prev_steps",
        type=int,
        default=5,
        help="Maximum number of previous steps to include (default: 5)"
    )
    
    args = parser.parse_args()
    
    prepare_training_data_filtered(
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
        max_prev_steps=args.max_prev_steps
    )
