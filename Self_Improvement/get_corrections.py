#!/usr/bin/env python3
"""
Step 2: Get GPT-5.2 Corrections for Model Errors

This script:
1. Loads error samples from Step 1
2. Sends them to GPT-5.2 for correct responses
3. Saves corrections for training

Usage:
    export OPENAI_API_KEY="your-key-here"
    
    python3 get_corrections.py \
        --error_file ./error_analysis.json \
        --data_dir ../SFT/training_data_filtered \
        --output_path ./corrections.jsonl
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from tqdm import tqdm


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image path."""
    ext = image_path.lower().split('.')[-1]
    return {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }.get(ext, 'image/jpeg')


def build_gpt_messages(prompt_text: str, image_paths: list, data_dir: str) -> list:
    """Build messages for GPT API with images."""
    content = []
    
    # Add images first
    for img_path in image_paths:
        full_path = os.path.join(data_dir, img_path)
        if os.path.exists(full_path):
            base64_img = encode_image_to_base64(full_path)
            media_type = get_image_media_type(full_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_img}",
                    "detail": "high"
                }
            })
    
    # Add text
    content.append({
        "type": "text",
        "text": prompt_text
    })
    
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="Get GPT-5.2 corrections for errors")
    parser.add_argument("--error_file", type=str, 
                        default="./error_analysis.json",
                        help="Path to error analysis from Step 1")
    parser.add_argument("--data_dir", type=str, 
                        default="../SFT/training_data_filtered",
                        help="Directory containing training data and images")
    parser.add_argument("--output_path", type=str, 
                        default="./corrections.jsonl",
                        help="Output path for corrections")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="GPT model to use")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens for GPT response")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 2: GET GPT-5.2 CORRECTIONS")
    print("=" * 60)
    print(f"Error file: {args.error_file}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    # Load error analysis
    print("\n[1/3] Loading error analysis...")
    
    with open(args.error_file, 'r') as f:
        error_data = json.load(f)
    
    errors = error_data['errors']
    print(f"✅ Loaded {len(errors)} error samples")
    
    # Load original training data to get full prompts
    print("\n[2/3] Loading original training data...")
    
    train_path = os.path.join(args.data_dir, "train.jsonl")
    original_samples = []
    with open(train_path, 'r') as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    print(f"✅ Loaded {len(original_samples)} original samples")
    
    # Check for resume
    processed_indices = set()
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_indices.add(data['sample_idx'])
        print(f"   Resuming: {len(processed_indices)} already processed")
    
    # Initialize OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Get corrections from GPT-5.2
    print("\n[3/3] Getting GPT-5.2 corrections...")
    
    total_input_tokens = 0
    total_output_tokens = 0
    successful = 0
    failed = 0
    
    with open(args.output_path, 'a' if args.resume else 'w') as f:
        for error in tqdm(errors, desc="Getting corrections"):
            sample_idx = error['sample_idx']
            
            # Skip if already processed
            if sample_idx in processed_indices:
                continue
            
            # Get original sample
            original = original_samples[sample_idx]
            
            # Extract prompt from original
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
            
            # Build GPT messages
            try:
                messages = build_gpt_messages(prompt_text, image_paths, args.data_dir)
                
                # Call GPT API
                api_params = {
                    "model": args.model,
                    "messages": messages,
                }
                
                # Use max_completion_tokens for newer models
                if "gpt-5" in args.model or "o1" in args.model or "o3" in args.model or "gpt-4o" in args.model:
                    api_params["max_completion_tokens"] = args.max_tokens
                else:
                    api_params["max_tokens"] = args.max_tokens
                
                response = client.chat.completions.create(**api_params)
                
                gpt_response = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Save correction
                correction = {
                    "sample_idx": sample_idx,
                    "original_expected": error['expected'],
                    "model_predicted": error['predicted'],
                    "gpt_correction": gpt_response,
                    "image_paths": image_paths,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                f.write(json.dumps(correction) + "\n")
                f.flush()
                
                successful += 1
                
            except Exception as e:
                print(f"\n❌ Error on sample {sample_idx}: {str(e)[:100]}")
                failed += 1
            
            # Rate limiting
            time.sleep(args.rate_limit_delay)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("CORRECTION COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    
    # Estimate cost (GPT-5.2 pricing estimate)
    input_cost = total_input_tokens / 1000 * 0.01  # Estimate
    output_cost = total_output_tokens / 1000 * 0.03  # Estimate
    print(f"Estimated cost: ${input_cost + output_cost:.2f}")
    
    print(f"\n✅ Corrections saved to {args.output_path}")


if __name__ == "__main__":
    main()

