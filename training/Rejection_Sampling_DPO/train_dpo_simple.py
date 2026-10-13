#!/usr/bin/env python3
"""
Simple DPO Training for VLMs

Historical reference-free implementation, not the agreed standard v9 DPO loss.
The canonical four-field dataset is deliberately rejected before loading a model.

This implements DPO loss directly, avoiding TRL's complex tokenization issues.
Uses SFT-style training but with DPO loss computation.
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from typing import List
from tqdm import tqdm
import math


def load_dpo_pairs(pairs_path: str) -> List[dict]:
    """Load DPO pairs."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    if any(isinstance(pair.get('prompt'), list) for pair in pairs):
        raise ValueError(
            'The frozen v9 dataset requires standard DPO with a frozen reference '
            'and complete response-token masking. This historical reference-free '
            'trainer is not compatible; no model has been loaded.'
        )
    return pairs


def compute_dpo_loss(
    model,
    chosen_input_ids,
    chosen_attention_mask,
    chosen_pixel_values,
    chosen_image_grid_thw,
    rejected_input_ids, 
    rejected_attention_mask,
    rejected_pixel_values,
    rejected_image_grid_thw,
    prompt_length,
    beta=0.1,
):
    """
    Compute DPO loss without needing reference model (simplified version).
    Uses implicit reference model assumption.
    """
    # Get logits for chosen (with image data for VLM)
    chosen_outputs = model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask,
        pixel_values=chosen_pixel_values,
        image_grid_thw=chosen_image_grid_thw,
    )
    chosen_logits = chosen_outputs.logits
    
    # Get logits for rejected (with image data for VLM)
    rejected_outputs = model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask,
        pixel_values=rejected_pixel_values,
        image_grid_thw=rejected_image_grid_thw,
    )
    rejected_logits = rejected_outputs.logits
    
    # Compute log probs for response tokens only (after prompt)
    # Shift logits and labels for next token prediction
    chosen_logits = chosen_logits[:, :-1, :]
    chosen_labels = chosen_input_ids[:, 1:]
    
    rejected_logits = rejected_logits[:, :-1, :]
    rejected_labels = rejected_input_ids[:, 1:]
    
    # Compute log probs
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    
    # Gather log probs for actual tokens
    chosen_token_log_probs = torch.gather(
        chosen_log_probs, 
        dim=-1, 
        index=chosen_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    rejected_token_log_probs = torch.gather(
        rejected_log_probs,
        dim=-1,
        index=rejected_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Only consider response tokens (after prompt)
    response_mask_chosen = torch.zeros_like(chosen_labels, dtype=torch.bool)
    response_mask_rejected = torch.zeros_like(rejected_labels, dtype=torch.bool)
    
    # Mask: 1 for response tokens, 0 for prompt tokens
    if prompt_length < chosen_labels.shape[1]:
        response_mask_chosen[:, prompt_length:] = True
    if prompt_length < rejected_labels.shape[1]:
        response_mask_rejected[:, prompt_length:] = True
    
    # Sum log probs for response only
    chosen_response_log_prob = (chosen_token_log_probs * response_mask_chosen.float()).sum(dim=-1)
    rejected_response_log_prob = (rejected_token_log_probs * response_mask_rejected.float()).sum(dim=-1)
    
    # DPO loss: -log(sigmoid(beta * (log_p_chosen - log_p_rejected)))
    log_ratio = chosen_response_log_prob - rejected_response_log_prob
    loss = -F.logsigmoid(beta * log_ratio).mean()
    
    return loss


def main():
    parser = argparse.ArgumentParser(description="Simple DPO Training")
    parser.add_argument("--base_model", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final")
    parser.add_argument("--pairs_path", type=str, required=True,
                        help="Explicit legacy input only; the canonical v9 set requires standard frozen-reference DPO")
    parser.add_argument("--data_dir", type=str, default="../SFT/training_data_filtered")
    parser.add_argument("--output_dir", type=str, default="./qwen3_vl_verifier_dpo_legacy")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE DPO TRAINING FOR VLM")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"DPO Pairs: {args.pairs_path}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print("=" * 60)
    
    # Load pairs
    print("\n[1/4] Loading DPO pairs...")
    pairs = load_dpo_pairs(args.pairs_path)
    print(f"✅ Loaded {len(pairs)} pairs")
    
    # Load model
    print("\n[2/4] Loading model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    # Check LoRA
    has_lora = hasattr(model, 'peft_config') or hasattr(model, 'active_adapter')
    if has_lora:
        print("  Model already has LoRA adapters")
    else:
        model = FastModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    
    model.train()
    # Note: gradient_checkpointing breaks 4-bit models, skip it
    print(f"✅ Model loaded")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Clear GPU cache before training
    torch.cuda.empty_cache()
    
    # Training
    print("\n[3/4] Training with DPO...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_steps = len(pairs) * args.epochs
    global_step = 0
    accumulated_loss = 0
    
    start_time = datetime.now()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        successful_samples = 0
        failed_samples = 0
        
        progress_bar = tqdm(pairs, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, pair in enumerate(progress_bar):
            try:
                # Load and resize images
                images = []
                for img_path in pair['image_paths']:
                    full_path = os.path.join(args.data_dir, img_path)
                    if os.path.exists(full_path):
                        img = Image.open(full_path).convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        images.append(img)
                
                if not images:
                    continue
                
                # Build conversations
                prompt_messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": pair['prompt']}]
                    }
                ]
                
                chosen_messages = prompt_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": pair['chosen']}]}
                ]
                
                rejected_messages = prompt_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": pair['rejected']}]}
                ]
                
                # Get text
                prompt_text = processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                chosen_text = processor.apply_chat_template(
                    chosen_messages, tokenize=False, add_generation_prompt=False
                )
                rejected_text = processor.apply_chat_template(
                    rejected_messages, tokenize=False, add_generation_prompt=False
                )
                
                # Tokenize - use processor properly for VLM
                prompt_inputs = processor(
                    text=[prompt_text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                prompt_length = prompt_inputs["input_ids"].shape[1]
                
                chosen_inputs = processor(
                    text=[chosen_text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                
                rejected_inputs = processor(
                    text=[rejected_text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                
                # Move to device and convert dtypes
                chosen_input_ids = chosen_inputs["input_ids"].to(model.device)
                chosen_attention_mask = chosen_inputs["attention_mask"].to(model.device)
                chosen_pixel_values = chosen_inputs.get("pixel_values")
                if chosen_pixel_values is not None:
                    chosen_pixel_values = chosen_pixel_values.to(model.device, dtype=torch.bfloat16)
                chosen_image_grid_thw = chosen_inputs.get("image_grid_thw")
                if chosen_image_grid_thw is not None:
                    chosen_image_grid_thw = chosen_image_grid_thw.to(model.device)
                
                rejected_input_ids = rejected_inputs["input_ids"].to(model.device)
                rejected_attention_mask = rejected_inputs["attention_mask"].to(model.device)
                rejected_pixel_values = rejected_inputs.get("pixel_values")
                if rejected_pixel_values is not None:
                    rejected_pixel_values = rejected_pixel_values.to(model.device, dtype=torch.bfloat16)
                rejected_image_grid_thw = rejected_inputs.get("image_grid_thw")
                if rejected_image_grid_thw is not None:
                    rejected_image_grid_thw = rejected_image_grid_thw.to(model.device)
                
                # Compute DPO loss with autocast for consistent dtypes
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = compute_dpo_loss(
                        model,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_pixel_values,
                        chosen_image_grid_thw,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_pixel_values,
                        rejected_image_grid_thw,
                        prompt_length,
                        beta=args.beta,
                    )
                
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum
                loss.backward()
                
                accumulated_loss += loss.item() * args.grad_accum
                epoch_loss += loss.item() * args.grad_accum
                successful_samples += 1
                
                # Gradient accumulation step
                if (i + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    avg_loss = accumulated_loss / args.grad_accum
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                    accumulated_loss = 0
                    
                    # Clear cache periodically to prevent OOM
                    if global_step % 10 == 0:
                        torch.cuda.empty_cache()
                
            except Exception as e:
                failed_samples += 1
                # Print first few errors to help debug
                if failed_samples <= 3:
                    print(f"\n  Error on sample {i}: {e}")
                # Clear memory on failure
                torch.cuda.empty_cache()
                continue
        
        # Final gradient step if needed
        if (len(pairs) % args.grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_epoch_loss = epoch_loss / max(successful_samples, 1)
        print(f"  Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Success: {successful_samples}/{len(pairs)}, Failed: {failed_samples}")
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # Save
    print("\n[4/4] Saving model...")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    
    print(f"✅ Model saved to {final_dir}")
    
    # Save info
    info = {
        "base_model": args.base_model,
        "pairs_used": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.lr,
        "training_time": str(training_time),
    }
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTest: python3 ../SFT/test_verifier.py --model_path {final_dir}")


if __name__ == "__main__":
    main()
