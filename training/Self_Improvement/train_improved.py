#!/usr/bin/env python3
"""
Step 3: Train Improved Model on Corrections

This script:
1. Loads the distilled model
2. Loads corrections from GPT-5.2
3. Creates training data from corrections
4. Fine-tunes the model on corrections

Usage:
    python3 train_improved.py \
        --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
        --corrections_path ./corrections.jsonl \
        --data_dir ../SFT/training_data_filtered \
        --output_dir ./qwen3_vl_verifier_improved_8b
"""

import argparse
import json
import os
from datetime import datetime
from PIL import Image
from typing import List, Dict


def load_corrections(corrections_path: str) -> List[dict]:
    """Load corrections from JSONL file."""
    corrections = []
    with open(corrections_path, 'r') as f:
        for line in f:
            corrections.append(json.loads(line))
    return corrections


def load_original_samples(data_dir: str) -> List[dict]:
    """Load original training samples."""
    samples = []
    train_path = os.path.join(data_dir, "train.jsonl")
    with open(train_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def prepare_training_sample(correction: dict, original_samples: list, data_dir: str) -> dict:
    """Prepare a training sample from correction."""
    sample_idx = correction['sample_idx']
    original = original_samples[sample_idx]
    
    # Extract original prompt
    prompt_text = ""
    image_paths = []
    for msg in original['messages']:
        if msg['role'] == 'user':
            content = msg['content']
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        prompt_text = item['text']
                    elif item.get('type') == 'image':
                        image_paths.append(item.get('image', ''))
    
    # Load images
    images = []
    for img_path in image_paths:
        full_path = os.path.join(data_dir, img_path)
        if os.path.exists(full_path):
            images.append(Image.open(full_path).convert("RGB"))
    
    # Build message format
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": correction['gpt_correction']}]
        }
    ]
    
    return {
        "messages": messages,
        "images": images,
        "sample_idx": sample_idx
    }


def main():
    parser = argparse.ArgumentParser(description="Train improved model on corrections")
    parser.add_argument("--base_model", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final",
                        help="Path to base model")
    parser.add_argument("--corrections_path", type=str, 
                        default="./corrections.jsonl",
                        help="Path to corrections from Step 2")
    parser.add_argument("--data_dir", type=str, 
                        default="../SFT/training_data_filtered",
                        help="Directory containing original training data and images")
    parser.add_argument("--output_dir", type=str, 
                        default="./qwen3_vl_verifier_improved_8b",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length")
    parser.add_argument("--include_original", action="store_true",
                        help="Include original correct samples in training (prevents forgetting)")
    parser.add_argument("--original_ratio", type=float, default=0.5,
                        help="Ratio of original samples to include if --include_original")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 3: TRAIN IMPROVED MODEL ON CORRECTIONS")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Corrections: {args.corrections_path}")
    print(f"Output: {args.output_dir}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Include Original: {args.include_original}")
    print("=" * 60)
    
    # Load corrections
    print("\n[1/4] Loading corrections...")
    
    corrections = load_corrections(args.corrections_path)
    print(f"✅ Loaded {len(corrections)} corrections")
    
    # Load original samples
    print("\n[2/4] Loading original training data...")
    
    original_samples = load_original_samples(args.data_dir)
    print(f"✅ Loaded {len(original_samples)} original samples")
    
    # Get error indices
    error_indices = set(c['sample_idx'] for c in corrections)
    
    # Prepare training data
    print("\n[3/4] Preparing training data...")
    
    training_data = []
    
    # Add correction samples
    for correction in corrections:
        sample = prepare_training_sample(correction, original_samples, args.data_dir)
        if sample['images']:  # Only add if images loaded successfully
            training_data.append(sample)
    
    print(f"  Added {len(training_data)} correction samples")
    
    # Optionally add original correct samples to prevent forgetting
    if args.include_original:
        import random
        correct_indices = [i for i in range(len(original_samples)) if i not in error_indices]
        num_to_add = int(len(corrections) * args.original_ratio)
        selected_indices = random.sample(correct_indices, min(num_to_add, len(correct_indices)))
        
        for idx in selected_indices:
            original = original_samples[idx]
            
            # Extract data from original sample
            prompt_text = ""
            response_text = ""
            image_paths = []
            
            for msg in original['messages']:
                if msg['role'] == 'user':
                    content = msg['content']
                    if isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'text':
                                prompt_text = item['text']
                            elif item.get('type') == 'image':
                                image_paths.append(item.get('image', ''))
                elif msg['role'] == 'assistant':
                    content = msg['content']
                    if isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'text':
                                response_text = item['text']
                    elif isinstance(content, str):
                        response_text = content
            
            # Load images
            images = []
            for img_path in image_paths:
                full_path = os.path.join(args.data_dir, img_path)
                if os.path.exists(full_path):
                    images.append(Image.open(full_path).convert("RGB"))
            
            if images:
                training_data.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response_text}]
                        }
                    ],
                    "images": images,
                    "sample_idx": idx
                })
        
        print(f"  Added {len(selected_indices)} original samples (to prevent forgetting)")
    
    print(f"  Total training samples: {len(training_data)}")
    
    # Shuffle training data
    import random
    random.shuffle(training_data)
    
    # Load model
    print("\n[4/4] Loading model and training...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    # Get LoRA model
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    
    print(f"✅ Model loaded with LoRA")
    
    # Prepare dataset
    from datasets import Dataset
    
    def process_sample(sample):
        text = processor.apply_chat_template(
            sample["messages"],
            images=sample["images"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text, "images": sample["images"]}
    
    processed_data = [process_sample(s) for s in training_data]
    dataset = Dataset.from_list(processed_data)
    
    # Train
    from trl import SFTTrainer, SFTConfig
    
    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        dataset_text_field="text",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=processor,
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
        "corrections_used": len(corrections),
        "total_training_samples": len(training_data),
        "include_original": args.include_original,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "training_time": str(training_time),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Test the improved model:")
    print(f"  cd ../SFT")
    print(f"  python3 test_verifier.py \\")
    print(f"      --model_path ../Self_Improvement/{args.output_dir}/final \\")
    print(f"      --data_dir ../testing_data_androidworld \\")
    print(f"      --val_file test.jsonl")


if __name__ == "__main__":
    main()

