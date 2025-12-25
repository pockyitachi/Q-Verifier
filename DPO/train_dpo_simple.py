#!/usr/bin/env python3
"""
Simple DPO Training for Verifier (Text-Only)

Uses TRL DPOTrainer with text-only data.
The model already has visual understanding from SFT/distillation.
DPO reinforces preference for correct Yes/No answers.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List


def load_dpo_pairs(pairs_path: str) -> List[dict]:
    """Load DPO preference pairs."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Simple DPO Training (Text-Only)")
    parser.add_argument("--base_model", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final",
                        help="Path to distilled model")
    parser.add_argument("--pairs_path", type=str, 
                        default="./dpo_pairs/dpo_pairs.jsonl",
                        help="Path to DPO pairs")
    parser.add_argument("--output_dir", type=str, 
                        default="./qwen3_vl_verifier_dpo_4b",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (lower for DPO)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE DPO TRAINING (TEXT-ONLY)")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"DPO Pairs: {args.pairs_path}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Load pairs
    print("\n[1/4] Loading DPO pairs...")
    pairs = load_dpo_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} preference pairs")
    
    # Load model with Unsloth
    print("\n[2/4] Loading model...")
    
    from unsloth import FastModel
    
    model, tokenizer = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
    )
    
    # Get the actual tokenizer if it's a processor
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer
    
    # Ensure pad token
    if actual_tokenizer.pad_token is None:
        actual_tokenizer.pad_token = actual_tokenizer.eos_token
    
    print(f"✅ Loaded model from {args.base_model}")
    
    # Prepare dataset - simplified text format
    print("\n[3/4] Preparing DPO dataset...")
    
    from datasets import Dataset
    
    dpo_data = []
    for pair in pairs:
        # Extract subgoal from prompt for shorter context
        prompt = pair.get('prompt', '')
        subgoal = pair.get('subgoal', '')
        
        # Create condensed prompt
        condensed_prompt = f"Evaluate this GUI transition.\nSubgoal: {subgoal}\nIs this transition valid?"
        
        dpo_data.append({
            "prompt": condensed_prompt,
            "chosen": pair['chosen'],
            "rejected": pair['rejected'],
        })
    
    dataset = Dataset.from_list(dpo_data)
    print(f"✅ Prepared {len(dataset)} DPO samples")
    
    # Train with DPO using standard TRL
    print("\n[4/4] Starting DPO training...")
    
    from trl import DPOTrainer, DPOConfig
    
    training_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        tokenizer=actual_tokenizer,
    )
    
    # Train
    start_time = datetime.now()
    trainer.train()
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # Save final model
    print("\nSaving final model...")
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    actual_tokenizer.save_pretrained(final_dir)
    
    print(f"✅ Model saved to {final_dir}")
    
    # Save training info
    info = {
        "base_model": args.base_model,
        "pairs_used": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.lr,
        "training_time": str(training_time),
        "completed_at": datetime.now().isoformat(),
        "note": "Text-only DPO training"
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

