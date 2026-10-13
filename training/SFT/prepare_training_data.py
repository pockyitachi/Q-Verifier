#!/usr/bin/env python3
"""
Prepare trajectory data for Qwen3-VL verifier training.

This script:
1. Collects all trajectory data from trajectory_data/
2. Copies images to a single folder with unique names
3. Updates image paths in the training samples
4. Outputs train.jsonl and val.jsonl files

Usage:
    python prepare_training_data.py --output_dir training_data --val_split 0.1
"""

import json
import os
import shutil
import argparse
import random
from pathlib import Path


def prepare_training_data(
    trajectory_dir: str = "trajectory_data",
    output_dir: str = "training_data",
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Prepare training data for verifier model.
    
    Args:
        trajectory_dir: Path to trajectory data folder
        output_dir: Path to output folder
        val_split: Fraction of data for validation (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PREPARING TRAINING DATA FOR VERIFIER")
    print("=" * 60)
    print(f"Source: {trajectory_dir}")
    print(f"Output: {output_dir}")
    print(f"Val split: {val_split * 100:.0f}%")
    print()
    
    all_samples = []
    total_images_copied = 0
    tasks_processed = 0
    
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
        if not messages:
            print(f"⚠️  Skipping {task_id}: No messages")
            continue
        
        # Process each sample
        for sample in messages:
            # Deep copy to avoid modifying original
            new_sample = json.loads(json.dumps(sample))
            
            # Update image paths in the sample
            if "messages" in new_sample:
                for msg in new_sample["messages"]:
                    if "content" in msg:
                        for content_item in msg["content"]:
                            if content_item.get("type") == "image":
                                old_path = content_item["image"]
                                # Extract filename from path like "images/1_before.jpg"
                                filename = os.path.basename(old_path)
                                # Create new unique filename
                                new_filename = f"{task_id}_{filename}"
                                new_path = f"images/{new_filename}"
                                
                                # Copy image if not already copied
                                src_image = task_dir / old_path
                                dst_image = images_path / new_filename
                                
                                if src_image.exists() and not dst_image.exists():
                                    shutil.copy2(src_image, dst_image)
                                    total_images_copied += 1
                                
                                # Update path in sample
                                content_item["image"] = new_path
            
            all_samples.append(new_sample)
        
        tasks_processed += 1
        print(f"✓ {task_id}: {len(messages)} samples")
    
    print()
    print(f"Processed {tasks_processed} tasks")
    print(f"Total samples: {len(all_samples)}")
    print(f"Images copied: {total_images_copied}")
    
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
    print("Done! Ready for Qwen3-VL training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for verifier")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectory_data",
        help="Path to trajectory data folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_data",
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
    
    args = parser.parse_args()
    
    prepare_training_data(
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed
    )
