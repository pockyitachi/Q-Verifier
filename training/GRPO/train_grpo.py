#!/usr/bin/env python3
"""
GRPO Training - Smart Reward Function

Based on error analysis of distilled model:
- 76% of errors are False Negatives (model too strict)
- 24% of errors are False Positives (model too lenient)

Reward function is designed to:
1. Penalize False Negatives more (asymmetric rewards)
2. Detect common error patterns in reasoning
3. Reward calibrated confidence

Usage:
    python3 train_grpo.py \
        --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
        --data_dir ../SFT/training_data_filtered \
        --output_dir ./qwen3_vl_verifier_grpo_8b
"""

import argparse
import json
import os
import re
from datetime import datetime
from PIL import Image
from typing import List, Dict
import torch


def extract_answer(response_text: str) -> str:
    """Extract Yes/No answer from response."""
    try:
        data = json.loads(response_text)
        return data.get("answer", "").lower()
    except:
        match = re.search(r'"answer"\s*:\s*"(yes|no)"', response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        text_lower = response_text.lower()
        if text_lower.startswith('{"answer": "yes"') or text_lower.startswith('{"answer":"yes"'):
            return "yes"
        elif text_lower.startswith('{"answer": "no"') or text_lower.startswith('{"answer":"no"'):
            return "no"
        if '"yes"' in text_lower[:100]:
            return "yes"
        elif '"no"' in text_lower[:100]:
            return "no"
    return "unknown"


def get_expected_answer(messages: list) -> str:
    """Extract expected answer from assistant message."""
    for msg in messages:
        if msg['role'] == 'assistant':
            content = msg['content']
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        return extract_answer(item['text'])
            elif isinstance(content, str):
                return extract_answer(content)
    return "unknown"


def get_prompt_content(messages: list) -> tuple:
    """Extract prompt text and image paths."""
    for msg in messages:
        if msg['role'] == 'user':
            content = msg['content']
            text_parts = []
            image_paths = []
            
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item['text'])
                    elif item.get('type') == 'image':
                        image_paths.append(item.get('image', ''))
            
            return '\n'.join(text_parts), image_paths
    return "", []


class SmartRewardFunction:
    """
    Smart reward function based on error analysis.
    
    Key insights from distilled model errors:
    - 76% False Negatives: Model says No when should be Yes
    - 24% False Positives: Model says Yes when should be No
    
    Design principles:
    1. ASYMMETRIC REWARDS: Penalize FN more than FP (since FN is bigger problem)
    2. PATTERN DETECTION: Detect phrases that indicate common error patterns
    3. CONFIDENCE CALIBRATION: Reward appropriate reasoning length
    """
    
    # Required by TRL's GRPOTrainer
    __name__ = "SmartRewardFunction"
    
    def __init__(self, log_path="./grpo_scoring_log.jsonl"):
        self.prompt_to_expected = {}
        self.log_path = log_path
        self.call_count = 0
        
        # Common phrases in False Negative errors (model over-analyzing)
        self.false_negative_patterns = [
            "does not immediately",
            "does not directly",
            "not aligned with",
            "does not move toward",
            "farther from the target",
            "no visible change",
            "did not change",
            "does not complete",
            "not the final",
        ]
        
        # Common phrases in False Positive errors
        # Model says Yes because "action ran" but doesn't check if it SUCCEEDED
        # Verifier must check BOTH: (1) right action? (2) did it work?
        self.false_positive_patterns = [
            "action is a direct step",
            "necessary step toward",
            "valid preparatory step",
            "moves the process toward",
            "is a step toward",
        ]
        
        # Phrases that indicate proper outcome checking (good)
        self.outcome_check_phrases = [
            "result",
            "outcome",
            "succeeded",
            "failed",
            "error",
            "not found",
            "did not",
            "still shows",
            "unchanged",
        ]
    
    def register_sample(self, prompt: str, expected: str):
        """Register expected answer for a prompt."""
        # Use hash of full prompt to avoid collisions
        import hashlib
        key = hashlib.md5(prompt.encode()).hexdigest()
        self.prompt_to_expected[key] = expected
    
    def detect_error_patterns(self, response: str, predicted: str) -> float:
        """
        Detect common error patterns and adjust reward.
        
        Returns adjustment to add to base reward.
        """
        response_lower = response.lower()
        adjustment = 0.0
        
        if predicted == "no":
            # If saying No, check for over-analysis patterns (common in FN)
            for pattern in self.false_negative_patterns:
                if pattern in response_lower:
                    # This pattern often leads to False Negatives
                    # Penalize slightly to discourage over-strict reasoning
                    adjustment -= 0.1
                    break
        
        elif predicted == "yes":
            # If saying Yes, check if reasoning verifies the OUTCOME
            # Verifier must check: (1) right action? AND (2) did it succeed?
            # Penalize if says Yes based only on "it's a step toward goal"
            # without checking if the action actually worked
            
            has_step_phrase = any(p in response_lower for p in self.false_positive_patterns)
            has_outcome_check = any(p in response_lower for p in self.outcome_check_phrases)
            
            if has_step_phrase and not has_outcome_check:
                # Says Yes because "it's a step toward goal" but doesn't verify outcome
                adjustment -= 0.2
        
        return adjustment
    
    def __call__(self, completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """
        Score completions with smart reward function.
        
        Asymmetric rewards:
        - Correct Yes: +2.5 (encourage saying Yes when correct, since FN is big problem)
        - Correct No:  +2.0 (standard reward)
        - Wrong Yes:   -2.0 (standard penalty)
        - Wrong No:    -2.5 (extra penalty, since FN is 76% of errors)
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            completion_str = str(completion)
            
            # Get expected answer
            expected = "unknown"
            if prompts and i < len(prompts):
                import hashlib
                key = hashlib.md5(str(prompts[i]).encode()).hexdigest()
                expected = self.prompt_to_expected.get(key, "unknown")
            
            predicted = extract_answer(completion_str)
            
            # === BASE REWARD: ASYMMETRIC TO ADDRESS ERROR DISTRIBUTION ===
            if expected != "unknown":
                if predicted == expected:
                    if predicted == "yes":
                        reward = 2.5   # Higher reward for correct Yes (reduce FN)
                    else:
                        reward = 2.0   # Standard reward for correct No
                elif predicted in ["yes", "no"]:
                    if predicted == "no":
                        reward = -2.5  # Higher penalty for wrong No (FN)
                    else:
                        reward = -2.0  # Standard penalty for wrong Yes (FP)
                else:
                    reward = -1.0  # Invalid format
            else:
                # Can't evaluate correctness
                if predicted in ["yes", "no"]:
                    reward = 0.0
                else:
                    reward = -0.5
            
            # === PATTERN-BASED ADJUSTMENT ===
            pattern_adj = self.detect_error_patterns(completion_str, predicted)
            reward += pattern_adj
            
            # === FORMAT BONUS (small) ===
            if '"answer"' in completion_str and '"reason"' in completion_str:
                reward += 0.1
            
            # === REASONING LENGTH (discourage too short or too long) ===
            # Good reasoning is typically 100-300 chars
            response_len = len(completion_str)
            if 100 <= response_len <= 400:
                reward += 0.1  # Good length
            elif response_len < 50:
                reward -= 0.2  # Too short (no reasoning)
            elif response_len > 600:
                reward -= 0.1  # Too long (over-analyzing)
            
            rewards.append(reward)
        
        # Log scoring details for analysis
        self.call_count += 1
        try:
            with open(self.log_path, "a") as f:
                for i, (completion, reward) in enumerate(zip(completions, rewards)):
                    completion_str = str(completion)
                    predicted = extract_answer(completion_str)
                    expected = "unknown"
                    if prompts and i < len(prompts):
                        import hashlib
                        key = hashlib.md5(str(prompts[i]).encode()).hexdigest()
                        expected = self.prompt_to_expected.get(key, "unknown")
                    
                    log_entry = {
                        "batch": self.call_count,
                        "idx": i,
                        "expected": expected,
                        "predicted": predicted,
                        "reward": reward,
                        "response_preview": completion_str[:300],
                    }
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            pass  # Don't break training for logging errors
        
        return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training - Smart Reward")
    parser.add_argument("--base_model", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final",
                        help="Path to base model")
    parser.add_argument("--data_dir", type=str, 
                        default="../SFT/training_data_filtered",
                        help="Directory containing training data")
    parser.add_argument("--data_file", type=str, default="train.jsonl",
                        help="Training data file")
    parser.add_argument("--output_dir", type=str, 
                        default="./qwen3_vl_verifier_grpo_8b",
                        help="Output directory")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRPO TRAINING - SMART REWARD FUNCTION")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Data: {args.data_dir}/{args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"Generations per prompt: {args.num_generations}")
    print()
    print("Reward Function (Asymmetric - addresses 76% FN error rate):")
    print("  Correct Yes:  +2.5  (encourage Yes when correct)")
    print("  Correct No:   +2.0")
    print("  Wrong Yes:    -2.0")
    print("  Wrong No:     -2.5  (extra penalty for FN)")
    print("  + Pattern detection adjustments")
    print("  + Reasoning length bonus/penalty")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    # Check if model already has LoRA adapters (from distillation)
    has_lora = hasattr(model, 'peft_config') or hasattr(model, 'active_adapter')
    
    if has_lora:
        print("  Model already has LoRA adapters - continuing training")
    else:
        model = FastModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    
    print(f"✅ Model loaded")
    
    # Load training data
    print("\n[2/4] Loading training data...")
    
    data_path = os.path.join(args.data_dir, args.data_file)
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Prepare dataset
    print("\n[3/4] Preparing GRPO dataset...")
    
    from datasets import Dataset
    
    scoring_log_path = os.path.join(args.output_dir, "scoring_log.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    reward_fn = SmartRewardFunction(log_path=scoring_log_path)
    grpo_data = []
    yes_count = 0
    no_count = 0
    
    for idx, sample in enumerate(samples):
        expected = get_expected_answer(sample['messages'])
        prompt_text, image_paths = get_prompt_content(sample['messages'])
        
        if expected == "yes":
            yes_count += 1
        elif expected == "no":
            no_count += 1
        
        # Load images (resize to reduce token count)
        images = []
        for img_path in image_paths:
            full_path = os.path.join(args.data_dir, img_path)
            if os.path.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                # Resize to reduce VLM token count (1440x3120 → 512x512)
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                images.append(img)
        
        if not images:
            continue
        
        # Build prompt
        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
            }
        ]
        
        prompt = processor.apply_chat_template(
            prompt_messages,
            images=images,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Register with reward function
        reward_fn.register_sample(prompt, expected)
        
        grpo_data.append({
            "prompt": prompt,
            "images": images,
        })
    
    dataset = Dataset.from_list(grpo_data)
    print(f"✅ Prepared {len(dataset)} samples")
    print(f"   Training distribution: Yes={yes_count}, No={no_count}")
    print(f"   Registered {len(reward_fn.prompt_to_expected)} prompts for reward")
    
    # Train with GRPO
    print("\n[4/4] Starting GRPO training...")
    print("  ⚠️  Note: TRL's GRPOTrainer may have issues with VLMs")
    print("      If it gets stuck, we may need a custom implementation like DPO")
    
    from trl import GRPOTrainer, GRPOConfig
    
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        max_completion_length=512,
        max_prompt_length=4096,  # Allow longer prompts for VLM
        temperature=0.8,
        remove_unused_columns=False,  # Important for VLM
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=processor,
    )
    
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
        "samples_used": len(grpo_data),
        "num_generations": args.num_generations,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "reward_type": "smart_asymmetric",
        "reward_correct_yes": 2.5,
        "reward_correct_no": 2.0,
        "reward_wrong_yes": -2.0,
        "reward_wrong_no": -2.5,
        "training_time": str(training_time),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("GRPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTest the model:")
    print(f"  cd ../SFT")
    print(f"  python3 test_verifier.py \\")
    print(f"      --model_path ../GRPO/{args.output_dir}/final \\")
    print(f"      --data_dir ../testing_data_androidworld \\")
    print(f"      --val_file test.jsonl")


if __name__ == "__main__":
    main()
