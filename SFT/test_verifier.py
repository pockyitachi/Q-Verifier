#!/usr/bin/env python3
"""
Test the trained Qwen3-VL Verifier on validation data.

Usage:
    python test_verifier.py --model_path ./qwen3_vl_verifier/final --num_samples 10
"""

import argparse
import json
import os
from datetime import datetime
from PIL import Image
from verifier_output import parse_verdict


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL Verifier")
    parser.add_argument("--model_path", type=str, default="./qwen3_vl_verifier/final",
                        help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="training_data_filtered",
                        help="Path to data directory")
    parser.add_argument("--val_file", type=str, default="val.jsonl",
                        help="Name of validation file (default: val.jsonl)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of validation samples to test")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for results (default: test_results_<model>.json)")
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output_file is None:
        model_name = os.path.basename(os.path.dirname(args.model_path))
        args.output_file = f"test_results_{model_name}.json"
    
    print("=" * 60)
    print("QWEN3-VL VERIFIER TEST")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Samples to test: {args.num_samples}")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading trained model...")
    
    from unsloth import FastModel
    
    model, processor = FastModel.from_pretrained(
        args.model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model.for_inference()  # Enable faster inference
    
    print(f"✅ Model loaded from {args.model_path}")
    
    # Load validation data
    print("\n[2/3] Loading validation data...")
    
    val_path = f"{args.data_dir}/{args.val_file}"
    val_samples = []
    with open(val_path, 'r') as f:
        for line in f:
            val_samples.append(json.loads(line))
    
    print(f"✅ Loaded {len(val_samples)} validation samples")
    
    # Test samples
    print(f"\n[3/3] Testing {args.num_samples} samples...")
    print("-" * 60)
    
    correct = 0
    total = 0
    results = []  # Store all results for saving
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, sample in enumerate(val_samples[:args.num_samples]):
        messages = sample["messages"]
        
        # Get user message and expected answer
        user_msg = messages[0]
        expected_answer = messages[1]["content"][0]["text"]
        
        # Build input with images
        images = []
        input_content = []
        image_idx = 0
        
        for item in user_msg["content"]:
            if item["type"] == "text":
                input_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                img_path = os.path.join(args.data_dir, item["image"])
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Resize to reasonable size to avoid token issues
                    max_size = 512
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    images.append(img)
                    # Use image with index for Qwen3-VL
                    input_content.append({"type": "image", "image": img})
                    image_idx += 1
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
        
        # Format for model - use the processor's apply_chat_template with images
        input_messages = [{"role": "user", "content": input_content}]
        
        # Use processor to apply chat template and process everything together
        inputs = processor.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        # Move to device
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Greedy for testing
            temperature=None,
            top_p=None,
        )
        
        # Count tokens
        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = outputs.shape[1] - input_token_count
        total_input_tokens += input_token_count
        total_output_tokens += output_token_count
        
        # Decode only generated tokens, retaining the full structured answer.
        response = processor.decode(outputs[0][input_token_count:], skip_special_tokens=True).strip()
        
        # Parse expected answer
        try:
            expected_json = json.loads(expected_answer)
            expected_verdict = expected_json.get("answer", "").lower()
        except:
            expected_verdict = "yes" if "yes" in expected_answer.lower()[:50] else "no"
        
        # Parse predicted answer
        predicted_verdict = parse_verdict(response)
        
        is_correct = predicted_verdict == expected_verdict
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        
        # Store result
        result_entry = {
            "sample_id": i + 1,
            "expected": expected_verdict,
            "predicted": predicted_verdict,
            "correct": is_correct,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "full_response": response,
            "expected_full": expected_answer,
        }
        results.append(result_entry)
        
        print(f"\nSample {i+1}/{args.num_samples} {status}")
        print(f"  Expected: {expected_verdict.upper()}")
        print(f"  Predicted: {predicted_verdict.upper()}")
        print(f"  Tokens: {input_token_count} in → {output_token_count} out")
        print(f"  Response preview: {response[:200]}...")
    
    # Calculate accuracy and token stats
    accuracy = 100 * correct / total if total > 0 else 0
    avg_input_tokens = total_input_tokens / total if total > 0 else 0
    avg_output_tokens = total_output_tokens / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"SR(Traj): {correct}/{total} ({accuracy:.1f}%)")
    print("-" * 60)
    print("TOKEN METRICS:")
    print(f"  Total Input Tokens:  {total_input_tokens:,}")
    print(f"  Total Output Tokens: {total_output_tokens:,}")
    print(f"  Avg Input/Sample:    {avg_input_tokens:.1f}")
    print(f"  Avg Output/Sample:   {avg_output_tokens:.1f}")
    print("=" * 60)
    
    # Save results to JSON
    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "num_samples": args.num_samples,
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
