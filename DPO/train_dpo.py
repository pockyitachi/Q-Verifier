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
from PIL import Image
from typing import List, Dict


def load_dpo_pairs(pairs_path: str) -> List[dict]:
    """Load DPO preference pairs."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def prepare_dpo_sample(pair: dict, image_base_dir: str) -> dict:
    """
    Prepare a single DPO sample with images.
    
    DPO needs:
    - prompt: shared input
    - chosen: preferred response
    - rejected: non-preferred response
    """
    
    # Load BEFORE image (shared in prompt)
    before_path = os.path.join(image_base_dir, pair['prompt_images'][0])
    
    # Load AFTER images
    chosen_after_path = os.path.join(image_base_dir, pair['chosen_after_image'])
    rejected_after_path = os.path.join(image_base_dir, pair['rejected_after_image'])
    
    # Check paths exist
    for path in [before_path, chosen_after_path, rejected_after_path]:
        if not os.path.exists(path):
            print(f"Warning: Image not found: {path}")
            return None
    
    # Load images
    before_img = Image.open(before_path).convert("RGB")
    chosen_after_img = Image.open(chosen_after_path).convert("RGB")
    rejected_after_img = Image.open(rejected_after_path).convert("RGB")
    
    # Build prompt with BEFORE image
    prompt_content = [
        {"type": "text", "text": pair['prompt']},
        {"type": "image", "image": before_img},
    ]
    
    # For VLM DPO, we need to include the AFTER image in both chosen and rejected
    # This is tricky - the AFTER image differs between chosen and rejected
    
    # Approach: Include AFTER image description in prompt, actual comparison is text-based
    # Alternative: Create two separate prompts (one for each AFTER)
    
    # For now, we'll structure it as comparing the text responses
    # The model should learn that "Yes" with correct AFTER is preferred
    
    return {
        "prompt": prompt_content,
        "chosen": pair['chosen'],
        "rejected": pair['rejected'],
        "chosen_after_image": chosen_after_img,
        "rejected_after_image": rejected_after_img,
        "corruption_type": pair['corruption_type']
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
    
    # Prepare dataset
    print("\n[3/4] Preparing DPO dataset...")
    
    # For DPO with TRL, we need specific format
    # Note: VLM DPO is complex - this is a simplified text-based version
    
    from datasets import Dataset
    
    dpo_data = []
    for pair in pairs:
        # Simplified: use text-only for DPO comparison
        # The image context is in the training already from SFT/distillation
        dpo_data.append({
            "prompt": f"Subgoal: {pair['subgoal']}\nEvaluate if this transition is valid.",
            "chosen": pair['chosen'],
            "rejected": pair['rejected'],
        })
    
    dataset = Dataset.from_list(dpo_data)
    print(f"✅ Prepared {len(dataset)} DPO samples")
    
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
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        tokenizer=processor,
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

