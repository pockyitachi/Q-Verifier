#!/usr/bin/env python3
"""
Step 4: DPO Training for Verifier

This script:
1. Loads the distilled model (Step 2 output)
2. Trains with DPO on preference pairs
3. Model learns to prefer correct transitions over corrupted ones

Usage:
    python train_dpo.py \
        --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final \
        --pairs_path ./dpo_pairs/dpo_pairs.jsonl \
        --output_dir ./qwen3_vl_verifier_dpo_4b
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
from PIL import Image


def load_dpo_pairs(pairs_path: str) -> List[dict]:
    """Load DPO preference pairs."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def prepare_dpo_sample_with_image_path(pair: dict, image_base_dir: str) -> dict:
    """
    Prepare DPO sample with image path for lazy loading.
    """
    prompt = pair['prompt']
    
    # Get image path
    before_image_path = os.path.join(image_base_dir, pair['prompt_images'][0])
    
    if not os.path.exists(before_image_path):
        return None
    
    # Load image - required for VLM DPO
    img = Image.open(before_image_path).convert("RGB")
    
    return {
        "prompt": prompt,
        "chosen": pair['chosen'],
        "rejected": pair['rejected'],
        "images": [img],
    }


def main():
    parser = argparse.ArgumentParser(description="DPO Training for Verifier")
    parser.add_argument("--base_model", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final",
                        help="Path to distilled model")
    parser.add_argument("--pairs_path", type=str, 
                        default="./dpo_pairs/dpo_pairs.jsonl",
                        help="Path to DPO pairs")
    parser.add_argument("--image_base_dir", type=str,
                        default="../SFT/training_data_filtered",
                        help="Base directory for images")
    parser.add_argument("--output_dir", type=str, 
                        default="./qwen3_vl_verifier_dpo_4b",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (DPO typically needs fewer)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (lower for DPO)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (controls preference strength)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 4: DPO TRAINING")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"DPO Pairs: {args.pairs_path}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Check if pairs exist
    if not os.path.exists(args.pairs_path):
        print(f"❌ ERROR: DPO pairs not found at {args.pairs_path}")
        print("   Run generate_dpo_pairs.py first!")
        return
    
    # Load pairs
    print("\n[1/4] Loading DPO pairs...")
    pairs = load_dpo_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} preference pairs")
    
    # Load model
    print("\n[2/4] Loading model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    print(f"✅ Loaded model from {args.base_model}")
    
    # Prepare dataset with images
    print("\n[3/4] Preparing DPO dataset with images...")
    
    from datasets import Dataset
    
    dpo_data = []
    skipped = 0
    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(pairs)}...")
        sample = prepare_dpo_sample_with_image_path(pair, args.image_base_dir)
        if sample is not None:
            dpo_data.append(sample)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} samples (missing images)")
    
    dataset = Dataset.from_list(dpo_data)
    print(f"✅ Prepared {len(dataset)} DPO samples with images")
    
    # Train with DPO
    print("\n[4/4] Starting DPO training...")
    
    from trl import DPOTrainer, DPOConfig
    
    training_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,  # DPO-specific parameter
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataset_num_proc=1,  # Single process for PIL image handling
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=processor,  # Unsloth expects the full processor
    )
    
    # Train
    start_time = datetime.now()
    trainer.train()
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # Save final model
    print("\nSaving final model...")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    
    print(f"✅ Model saved to {final_dir}")
    
    # Save training info
    info = {
        "base_model": args.base_model,
        "pairs_used": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.lr,
        "training_time": str(training_time),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

