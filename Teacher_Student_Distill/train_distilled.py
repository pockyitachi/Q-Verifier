#!/usr/bin/env python3
"""
Step 2: Train Student Model on GPT Distillation Data

This script:
1. Loads the Step 1 fine-tuned Qwen model
2. Continues training on GPT-generated responses
3. Saves the improved Step 2 model

Usage:
    python step2_train_distilled.py \
        --base_model ./qwen3_vl_verifier/final \
        --data_dir step2_gpt_data \
        --output_dir ./qwen3_vl_verifier_step2
"""

import argparse
import json
import os
from datetime import datetime
from PIL import Image


def load_sample(sample: dict, data_dir: str, image_base_dir: str) -> dict:
    """Load a sample with images for VLM training."""
    messages = sample["messages"]
    
    # Process user message to load images
    user_content = messages[0]["content"]
    processed_content = []
    
    for item in user_content:
        if item["type"] == "text":
            processed_content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            # Try both data_dir and image_base_dir for image path
            image_path = os.path.join(image_base_dir, item["image"])
            if not os.path.exists(image_path):
                image_path = os.path.join(data_dir, item["image"])
            
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                processed_content.append({"type": "image", "image": img})
            else:
                print(f"Warning: Image not found: {item['image']}")
    
    return {
        "messages": [
            {"role": "user", "content": processed_content},
            messages[1],  # Assistant response (GPT's answer)
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Step 2: Train on GPT distillation data")
    parser.add_argument("--base_model", type=str, default="../SFT/qwen3_vl_verifier/final",
                        help="Path to SFT trained model")
    parser.add_argument("--data_dir", type=str, default="./gpt_data",
                        help="Directory with GPT-generated training data")
    parser.add_argument("--image_base_dir", type=str, default="../SFT/training_data_filtered",
                        help="Base directory for images")
    parser.add_argument("--output_dir", type=str, default="./qwen3_vl_verifier_distilled",
                        help="Output directory for Step 2 model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 2: DISTILLATION TRAINING")
    print("=" * 60)
    print(f"Base Model (Step 1): {args.base_model}")
    print(f"Distillation Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Load Step 1 model
    print("\n[1/4] Loading SFT trained model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    print(f"✅ Loaded model from {args.base_model}")
    
    # Load distillation data
    print("\n[2/4] Loading distillation data...")
    
    # Try corrected file first, fallback to original
    train_path = os.path.join(args.data_dir, "train_distilled_corrected.jsonl")
    if not os.path.exists(train_path):
        train_path = os.path.join(args.data_dir, "train_distilled.jsonl")
    if not os.path.exists(train_path):
        print(f"❌ ERROR: No training data found in {args.data_dir}!")
        print("   Run generate_gpt_data.py first to generate GPT responses.")
        return
    print(f"  Using: {train_path}")
    
    train_samples = []
    with open(train_path, 'r') as f:
        for line in f:
            train_samples.append(json.loads(line))
    
    print(f"✅ Loaded {len(train_samples)} distillation samples")
    
    # Prepare dataset for VLM training
    print("\n[3/4] Preparing dataset...")
    
    train_data = [
        load_sample(s, args.data_dir, args.image_base_dir) 
        for s in train_samples
    ]
    
    print(f"✅ Prepared {len(train_data)} training samples")
    
    # Setup training
    print("\n[4/4] Starting training...")
    
    from trl import SFTTrainer, SFTConfig
    from unsloth import UnslothVisionDataCollator
    
    # Training config
    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        # VLM specific
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    # Data collator for VLM
    vlm_data_collator = UnslothVisionDataCollator(model, processor)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_data,
        data_collator=vlm_data_collator,
        processing_class=processor,
    )
    
    # Train
    print("\n🚀 Training started...")
    print("-" * 60)
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving Step 2 model...")
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    # Save training metadata
    metadata = {
        "step": 2,
        "description": "Teacher-Student Distillation (GPT → Qwen)",
        "base_model": args.base_model,
        "distillation_data": args.data_dir,
        "num_samples": len(train_samples),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(args.output_dir, "step2_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("STEP 2 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {final_path}")
    print("\nNext steps:")
    print("  1. Test Step 2 model:")
    print(f"     python test_verifier.py --model_path {final_path}")
    print("  2. Compare with Step 1 model")
    print("  3. Proceed to Step 3 (negative data) or Step 4 (DPO)")
    print("=" * 60)


if __name__ == "__main__":
    main()

