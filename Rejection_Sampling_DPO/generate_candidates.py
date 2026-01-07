#!/usr/bin/env python3
"""
Step 1: Generate Multiple Candidate Responses

This script:
1. Loads the model and training data
2. Generates N responses per sample with temperature sampling
3. Checks which responses are correct vs wrong
4. Saves candidates for DPO pair creation

Usage:
    python3 generate_candidates.py \
        --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
        --data_dir ../SFT/training_data_filtered \
        --num_responses 5 \
        --output_path ./candidate_responses.jsonl
"""

import argparse
import json
import os
import re
from datetime import datetime
from PIL import Image
from tqdm import tqdm


def extract_answer(response_text: str) -> str:
    """Extract Yes/No answer from response."""
    try:
        data = json.loads(response_text)
        return data.get("answer", "").lower()
    except:
        match = re.search(r'"answer"\s*:\s*"(yes|no)"', response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        if "yes" in response_text.lower()[:50]:
            return "yes"
        elif "no" in response_text.lower()[:50]:
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
    """Extract prompt text and image paths from user message."""
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


def main():
    parser = argparse.ArgumentParser(description="Generate candidate responses for DPO")
    parser.add_argument("--model_path", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final",
                        help="Path to model")
    parser.add_argument("--data_dir", type=str, 
                        default="../SFT/training_data_filtered",
                        help="Directory containing training data")
    parser.add_argument("--data_file", type=str, default="train.jsonl",
                        help="Training data file")
    parser.add_argument("--output_path", type=str, 
                        default="./candidate_responses.jsonl",
                        help="Output path for candidates")
    parser.add_argument("--num_responses", type=int, default=5,
                        help="Number of responses to generate per sample")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 1: GENERATE CANDIDATE RESPONSES")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}/{args.data_file}")
    print(f"Responses per sample: {args.num_responses}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.model_path,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)
    
    print(f"✅ Model loaded")
    
    # Load training data
    print("\n[2/3] Loading training data...")
    
    data_path = os.path.join(args.data_dir, args.data_file)
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Check for resume
    processed_indices = set()
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_indices.add(data['sample_idx'])
        print(f"   Resuming: {len(processed_indices)} already processed")
    
    # Generate candidates
    print("\n[3/3] Generating candidate responses...")
    
    stats = {"total": 0, "with_both": 0, "only_correct": 0, "only_wrong": 0}
    
    with open(args.output_path, 'a' if args.resume else 'w') as f:
        for idx, sample in enumerate(tqdm(samples, desc="Generating")):
            if idx in processed_indices:
                continue
            
            expected = get_expected_answer(sample['messages'])
            prompt_text, image_paths = get_prompt_content(sample['messages'])
            
            # Load images
            images = []
            for img_path in image_paths:
                full_path = os.path.join(args.data_dir, img_path)
                if os.path.exists(full_path):
                    images.append(Image.open(full_path).convert("RGB"))
            
            if not images:
                continue
            
            # Build conversation
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
                }
            ]
            
            # Process input - handle VLM properly
            text = processor.apply_chat_template(
                conversation,
                images=images,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate N responses with sampling
            responses = []
            correct_responses = []
            wrong_responses = []
            
            for _ in range(args.num_responses):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                )
                
                input_len = inputs["input_ids"].shape[1]
                response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
                predicted = extract_answer(response)
                
                response_data = {
                    "response": response,
                    "predicted": predicted,
                    "correct": (predicted == expected)
                }
                responses.append(response_data)
                
                if predicted == expected:
                    correct_responses.append(response)
                elif predicted in ["yes", "no"]:
                    wrong_responses.append(response)
            
            # Save result
            result = {
                "sample_idx": idx,
                "expected": expected,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "responses": responses,
                "num_correct": len(correct_responses),
                "num_wrong": len(wrong_responses),
                "correct_responses": correct_responses,
                "wrong_responses": wrong_responses,
                "has_both": len(correct_responses) > 0 and len(wrong_responses) > 0
            }
            
            f.write(json.dumps(result) + "\n")
            f.flush()
            
            # Update stats
            stats["total"] += 1
            if result["has_both"]:
                stats["with_both"] += 1
            elif len(correct_responses) > 0:
                stats["only_correct"] += 1
            else:
                stats["only_wrong"] += 1
    
    print(f"\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"With both correct & wrong: {stats['with_both']} ← Can create DPO pairs")
    print(f"Only correct responses: {stats['only_correct']}")
    print(f"Only wrong responses: {stats['only_wrong']}")
    print(f"\n✅ Saved to {args.output_path}")
    print(f"\nNext step: python3 create_dpo_pairs.py --candidates_path {args.output_path}")


if __name__ == "__main__":
    main()

