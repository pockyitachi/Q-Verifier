#!/usr/bin/env python3
"""
Train Qwen VL Verifier with Unsloth

Supports:
- Qwen3-VL: 4B, 8B
- Qwen2.5-VL: 3B, 7B

Usage:
    pip install unsloth unsloth_zoo datasets trl pillow
    python train_verifier.py --model unsloth/Qwen2.5-VL-3B-Instruct --output_dir ./qwen25_vl_verifier_3b
    python train_verifier.py --model unsloth/Qwen2.5-VL-7B-Instruct --output_dir ./qwen25_vl_verifier_7b
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Train Qwen VL Verifier (Qwen3-VL or Qwen2.5-VL)")
    parser.add_argument("--data_dir", type=str, default="training_data_filtered",
                        help="Path to training data directory")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-VL-4B-Instruct",
                        choices=[
                            "unsloth/Qwen3-VL-4B-Instruct", 
                            "unsloth/Qwen3-VL-8B-Instruct",
                            "unsloth/Qwen2.5-VL-3B-Instruct",
                            "unsloth/Qwen2.5-VL-7B-Instruct",
                        ],
                        help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="./qwen3_vl_verifier",
                        help="Output directory for checkpoints")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QWEN VL VERIFIER TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # =========================================
    # Step 1: Load Model
    # =========================================
    print("\n[1/5] Loading model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    print(f"✅ Loaded {args.model}")
    
    # =========================================
    # Step 2: Configure LoRA
    # =========================================
    print("\n[2/5] Configuring LoRA...")
    
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"✅ LoRA configured (r={args.lora_r})")
    
    # =========================================
    # Step 3: Load Dataset
    # =========================================
    print("\n[3/5] Loading dataset...")
    
    from datasets import Dataset
    
    def load_jsonl(filepath):
        samples = []
        with open(filepath, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples
    
    def convert_sample_for_vlm(sample, data_dir):
        """
        Convert sample to VLM training format.
        TRL SFTTrainer expects: {"messages": [...], "images": [PIL.Image, ...]}
        """
        messages = sample["messages"]
        images = []
        converted_messages = []
        
        for msg in messages:
            new_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    new_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    # Load image and add to images list
                    img_path = os.path.join(data_dir, item["image"])
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                        # Reference image by index
                        new_content.append({"type": "image"})
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}: {e}")
                        continue
            
            converted_messages.append({
                "role": msg["role"],
                "content": new_content
            })
        
        return {
            "messages": converted_messages,
            "images": images
        }
    
    # Load data
    train_data = load_jsonl(f"{args.data_dir}/train.jsonl")
    val_data = load_jsonl(f"{args.data_dir}/val.jsonl")
    
    print(f"  Raw train samples: {len(train_data)}")
    print(f"  Raw val samples: {len(val_data)}")
    
    # Convert to VLM format using list comprehension (as per Unsloth docs)
    # https://docs.unsloth.ai/models/qwen3-vl-how-to-run-and-fine-tune#fine-tuning-qwen3-vl
    # "Using map kicks in dataset standardization and arrow processing rules"
    print("  Converting to VLM format (list comprehension per Unsloth docs)...")
    train_list = [convert_sample_for_vlm(s, args.data_dir) for s in train_data]
    val_list = [convert_sample_for_vlm(s, args.data_dir) for s in val_data]
    
    # Filter out samples with no images
    train_list = [s for s in train_list if len(s["images"]) > 0]
    val_list = [s for s in val_list if len(s["images"]) > 0]
    
    print(f"  Train samples (with images): {len(train_list)}")
    print(f"  Val samples (with images): {len(val_list)}")
    
    # Use list directly instead of Dataset.from_list() to avoid arrow processing issues
    train_dataset = train_list
    val_dataset = val_list
    
    print(f"✅ Dataset loaded")
    
    # =========================================
    # Step 4: Setup Trainer
    # =========================================
    print("\n[4/5] Setting up trainer...")
    
    from trl import SFTTrainer, SFTConfig
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    
    # Create Unsloth's VLM data collator (required - TRL's collator gets overridden)
    vlm_data_collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        max_seq_length=args.max_seq_length,
    )
    
    # Use SFTConfig for VLM training
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        
        # Batch size
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        # Training
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        
        # Saving
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Mixed precision
        bf16=True,
        
        # Other
        seed=42,
        report_to="none",
        
        # VLM specific - don't try to tokenize, let the collator handle it
        dataset_text_field=None,
        max_seq_length=args.max_seq_length,
        
        # Resume
        resume_from_checkpoint=args.resume_from,
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=vlm_data_collator,
        args=sft_config,
    )
    
    print(f"✅ Trainer configured")
    
    # =========================================
    # Step 5: Train
    # =========================================
    print("\n[5/5] Starting training...")
    print("-" * 60)
    
    trainer.train(resume_from_checkpoint=args.resume_from)
    
    print("-" * 60)
    print("✅ Training complete!")
    
    # =========================================
    # Save Final Model
    # =========================================
    print("\nSaving model...")
    
    final_dir = f"{args.output_dir}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    
    print(f"✅ Model saved to {final_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {final_dir}")
    print(f"Checkpoints in: {args.output_dir}")
    print("\nTo use the model:")
    print(f"  model = FastModel.from_pretrained('{final_dir}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
