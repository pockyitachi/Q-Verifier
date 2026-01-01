#!/usr/bin/env python3
"""
Test GPT-5.2 (or GPT-4o) on validation data for comparison with fine-tuned models.

Usage:
    export OPENAI_API_KEY="sk-your-key"
    python test_gpt.py --num_samples 100 --model gpt-4o
    python test_gpt.py --num_samples 1082 --model gpt-5.2
"""

import argparse
import json
import os
import base64
import time
from datetime import datetime
from PIL import Image
from io import BytesIO
from openai import OpenAI


def encode_image_to_base64(image_path: str, max_size: int = 512) -> str:
    """Load image, resize if needed, and encode to base64."""
    img = Image.open(image_path).convert("RGB")
    
    # Resize to reasonable size to reduce token usage
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test GPT on validation data")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use (gpt-4o, gpt-4o-mini, gpt-5.2)")
    parser.add_argument("--data_dir", type=str, default="../testing_data_androidworld",
                        help="Path to data directory")
    parser.add_argument("--val_file", type=str, default="test.jsonl",
                        help="Name of validation file")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of validation samples to test")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index (for resuming)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Run: export OPENAI_API_KEY='sk-your-key'")
        return
    
    # Determine output filename
    if args.output_file is None:
        model_name = args.model.replace("-", "_").replace(".", "_")
        args.output_file = f"test_results_androidworld_{model_name}.json"
    
    print("=" * 60)
    print("GPT MODEL TEST")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}/{args.val_file}")
    print(f"Samples to test: {args.num_samples}")
    print(f"Starting from index: {args.start_idx}")
    print("=" * 60)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load validation data
    print("\n[1/2] Loading validation data...")
    
    val_path = f"{args.data_dir}/{args.val_file}"
    val_samples = []
    with open(val_path, 'r') as f:
        for line in f:
            val_samples.append(json.loads(line))
    
    print(f"✅ Loaded {len(val_samples)} validation samples")
    
    # Test samples
    print(f"\n[2/2] Testing {args.num_samples} samples with {args.model}...")
    print("-" * 60)
    
    correct = 0
    total = 0
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    end_idx = min(args.start_idx + args.num_samples, len(val_samples))
    
    for i, sample in enumerate(val_samples[args.start_idx:end_idx]):
        sample_idx = args.start_idx + i
        messages = sample["messages"]
        
        # Get user message and expected answer
        user_msg = messages[0]
        expected_answer = messages[1]["content"][0]["text"]
        
        # Build content for GPT API
        gpt_content = []
        
        for item in user_msg["content"]:
            if item["type"] == "text":
                gpt_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                img_path = os.path.join(args.data_dir, item["image"])
                try:
                    base64_image = encode_image_to_base64(img_path)
                    gpt_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"  # Use low detail to reduce tokens
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
        
        # Call GPT API
        try:
            # GPT-5.2 and newer models use max_completion_tokens
            # Older models use max_tokens
            api_params = {
                "model": args.model,
                "messages": [{"role": "user", "content": gpt_content}],
                "temperature": 0,  # Deterministic for testing
            }
            
            if "gpt-5" in args.model or "o1" in args.model or "o3" in args.model:
                api_params["max_completion_tokens"] = args.max_tokens
            else:
                api_params["max_tokens"] = args.max_tokens
            
            response = client.chat.completions.create(**api_params)
            
            gpt_response = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
        except Exception as e:
            print(f"❌ API Error on sample {sample_idx + 1}: {e}")
            results.append({
                "sample_id": sample_idx + 1,
                "error": str(e),
                "expected": None,
                "predicted": None,
                "correct": False,
            })
            total += 1
            time.sleep(args.delay * 2)  # Extra delay on error
            continue
        
        # Parse expected answer
        try:
            expected_json = json.loads(expected_answer)
            expected_verdict = expected_json.get("answer", "").lower()
        except:
            expected_verdict = "yes" if "yes" in expected_answer.lower()[:50] else "no"
        
        # Parse predicted answer
        predicted_verdict = "yes" if "yes" in gpt_response.lower()[:100] else "no"
        
        is_correct = predicted_verdict == expected_verdict
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        
        # Store result
        result_entry = {
            "sample_id": sample_idx + 1,
            "expected": expected_verdict,
            "predicted": predicted_verdict,
            "correct": is_correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "full_response": gpt_response,
            "expected_full": expected_answer,
        }
        results.append(result_entry)
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0:
            accuracy_so_far = 100 * correct / total if total > 0 else 0
            print(f"Sample {sample_idx + 1}/{end_idx} {status} | "
                  f"Running accuracy: {correct}/{total} ({accuracy_so_far:.1f}%) | "
                  f"Tokens: {input_tokens}→{output_tokens}")
        
        # Rate limiting delay
        time.sleep(args.delay)
    
    # Calculate final stats
    accuracy = 100 * correct / total if total > 0 else 0
    avg_input_tokens = total_input_tokens / total if total > 0 else 0
    avg_output_tokens = total_output_tokens / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"GPT TEST RESULTS ({args.model})")
    print("=" * 60)
    print(f"SR(Traj): {correct}/{total} ({accuracy:.1f}%)")
    print("-" * 60)
    print("TOKEN METRICS:")
    print(f"  Total Input Tokens:  {total_input_tokens:,}")
    print(f"  Total Output Tokens: {total_output_tokens:,}")
    print(f"  Avg Input/Sample:    {avg_input_tokens:.1f}")
    print(f"  Avg Output/Sample:   {avg_output_tokens:.1f}")
    print("=" * 60)
    
    # Estimate cost (approximate pricing)
    # GPT-4o: $2.50/1M input, $10/1M output
    # GPT-5.2: pricing may vary
    if "gpt-4o" in args.model:
        input_cost = (total_input_tokens / 1_000_000) * 2.50
        output_cost = (total_output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + output_cost
        print(f"\n💰 Estimated cost: ${total_cost:.2f}")
    
    # Save results to JSON
    output_data = {
        "metadata": {
            "model": args.model,
            "data_dir": args.data_dir,
            "val_file": args.val_file,
            "num_samples": args.num_samples,
            "start_idx": args.start_idx,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "sr_traj": {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            },
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
                "avg_input_per_sample": avg_input_tokens,
                "avg_output_per_sample": avg_output_tokens,
            },
        },
        "results": results,
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

