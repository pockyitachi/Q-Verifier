#!/usr/bin/env python3
"""
Step 2: Teacher-Student Distillation
Generate high-quality verification responses using GPT-4V (Teacher).

This script:
1. Loads training data samples
2. Sends them to GPT-4V/4o with images
3. Collects GPT's reasoning and answers
4. Saves as new training data for distillation

Usage:
    python step2_generate_gpt_data.py --api_key YOUR_KEY --num_samples 100
    
    # Or use environment variable
    export OPENAI_API_KEY=your_key
    python step2_generate_gpt_data.py --num_samples 100
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for GPT-4V API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image path."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def call_gpt4v(api_key: str, messages: list, model: str = "gpt-4o") -> dict:
    """Call GPT-4V/4o API with vision capabilities."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # GPT-5.2 uses max_completion_tokens, older models use max_tokens
    if model.startswith("gpt-5"):
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": 1024,
            "temperature": 0.3,
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    
    if response.status_code != 200:
        error_detail = response.json() if response.text else {"error": response.text}
        raise Exception(f"API Error {response.status_code}: {error_detail}")
    return response.json()


def build_gpt_messages(sample: dict, data_dir: str) -> list:
    """Build GPT-4V compatible messages from a training sample."""
    user_content = sample["messages"][0]["content"]
    
    gpt_content = []
    
    for item in user_content:
        if item["type"] == "text":
            # Enhance the prompt for GPT to generate better reasoning
            enhanced_text = item["text"] + """

IMPORTANT CLARIFICATION:
A subgoal may require MULTIPLE actions to complete. You are evaluating whether 
THIS SINGLE ACTION is a valid STEP TOWARDS the subgoal, NOT whether it fully 
completes the subgoal.

An action is VALID (Yes) if:
- It makes progress toward the subgoal
- It is a necessary preparatory step (e.g., going to home screen before opening an app)
- It contributes to eventually achieving the subgoal

An action is INVALID (No) if:
- It goes in the wrong direction
- It is counterproductive to the subgoal
- It does not contribute to progress (e.g., the expected change did not happen)

Provide a detailed, step-by-step analysis:
1. Describe what you see in the BEFORE image
2. Describe what you see in the AFTER image  
3. Explain what changed between the two states
4. Evaluate if this action is a valid STEP TOWARDS the subgoal (not whether it completes it)
5. Give your final answer with clear reasoning

Format your response as JSON:
{"answer": "Yes" or "No", "reason": "Your detailed reasoning here"}"""
            gpt_content.append({"type": "text", "text": enhanced_text})
            
        elif item["type"] == "image":
            image_path = os.path.join(data_dir, item["image"])
            if os.path.exists(image_path):
                base64_img = encode_image_to_base64(image_path)
                media_type = get_image_media_type(image_path)
                gpt_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_img}",
                        "detail": "high",
                    }
                })
    
    return [{"role": "user", "content": gpt_content}]


def main():
    parser = argparse.ArgumentParser(description="Generate GPT-4V distillation data")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        choices=["gpt-5.2", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                        help="GPT model to use")
    parser.add_argument("--data_dir", type=str, default="../SFT/training_data_filtered",
                        help="Path to training data directory")
    parser.add_argument("--input_file", type=str, default="train.jsonl",
                        help="Input JSONL file (train.jsonl or val.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./gpt_data",
                        help="Output directory for GPT responses")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index (for resuming)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OpenAI API key required!")
        print("   Set via --api_key or OPENAI_API_KEY environment variable")
        return
    
    print("=" * 60)
    print("STEP 2: GPT-4V DISTILLATION DATA GENERATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}/{args.input_file}")
    print(f"Samples: {args.num_samples} (starting at {args.start_idx})")
    print(f"Output: {args.output_dir}/")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    input_path = os.path.join(args.data_dir, args.input_file)
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"\n✅ Loaded {len(samples)} samples from {input_path}")
    
    # Select samples to process
    end_idx = min(args.start_idx + args.num_samples, len(samples))
    samples_to_process = samples[args.start_idx:end_idx]
    
    print(f"📝 Processing samples {args.start_idx} to {end_idx - 1}")
    print("-" * 60)
    
    # Output files
    output_jsonl = os.path.join(args.output_dir, f"gpt_responses_{args.start_idx}_{end_idx}.jsonl")
    output_train = os.path.join(args.output_dir, "train_distilled.jsonl")
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, sample in enumerate(samples_to_process):
        global_idx = args.start_idx + i
        print(f"\n[{i+1}/{len(samples_to_process)}] Processing sample {global_idx}...")
        
        try:
            # Build GPT messages
            gpt_messages = build_gpt_messages(sample, args.data_dir)
            
            # Call GPT-4V
            response = call_gpt4v(api_key, gpt_messages, args.model)
            
            gpt_answer = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            
            # Create result entry
            result = {
                "sample_idx": global_idx,
                "original_answer": sample["messages"][1]["content"][0]["text"],
                "gpt_answer": gpt_answer,
                "gpt_model": args.model,
                "tokens": {
                    "prompt": usage.get("prompt_tokens", 0),
                    "completion": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)
            
            # Also create distilled training sample
            distilled_sample = {
                "messages": [
                    sample["messages"][0],  # Keep original user message
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": gpt_answer}]
                    }
                ]
            }
            
            # Append to train file
            with open(output_train, 'a') as f:
                f.write(json.dumps(distilled_sample) + "\n")
            
            success_count += 1
            print(f"   ✅ Success | Tokens: {usage.get('total_tokens', 'N/A')}")
            
            # Preview answer
            preview = gpt_answer[:100].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append({
                "sample_idx": global_idx,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
        
        # Rate limiting
        if i < len(samples_to_process) - 1:
            time.sleep(args.delay)
    
    # Save detailed results
    with open(output_jsonl, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Success: {success_count}/{len(samples_to_process)}")
    print(f"Errors: {error_count}")
    print("-" * 60)
    print(f"Detailed results: {output_jsonl}")
    print(f"Training data: {output_train}")
    print("=" * 60)
    
    # Calculate cost estimate
    total_input = sum(r.get("tokens", {}).get("prompt", 0) for r in results)
    total_output = sum(r.get("tokens", {}).get("completion", 0) for r in results)
    total_tokens = total_input + total_output
    # GPT-5.2 pricing: $1.75/1M input, $14/1M output
    # GPT-4o pricing: $2.50/1M input, $10/1M output
    # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
    if args.model == "gpt-5.2":
        cost_estimate = (total_input * 1.75 + total_output * 14.0) / 1_000_000
    elif args.model == "gpt-4o-mini":
        cost_estimate = (total_input * 0.15 + total_output * 0.60) / 1_000_000
    else:  # gpt-4o, gpt-4-turbo
        cost_estimate = (total_input * 2.50 + total_output * 10.0) / 1_000_000
    print(f"\n💰 Estimated cost: ${cost_estimate:.2f} (approx)")
    print(f"   Total tokens used: {total_tokens:,}")


if __name__ == "__main__":
    main()

