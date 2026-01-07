#!/usr/bin/env python3
"""
Step 1: Find Model Errors on Training Data

This script:
1. Runs the distilled model on training data
2. Compares predictions to ground truth
3. Identifies samples where the model makes errors
4. Saves error samples for correction

Usage:
    python3 find_errors.py \
        --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
        --data_dir ../SFT/training_data_filtered \
        --output_path ./error_analysis.json
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
        # Try to parse as JSON
        data = json.loads(response_text)
        return data.get("answer", "").lower()
    except:
        # Fallback to regex
        match = re.search(r'"answer"\s*:\s*"(yes|no)"', response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        # Check for plain yes/no
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
    parser = argparse.ArgumentParser(description="Find model errors on training data")
    parser.add_argument("--model_path", type=str, 
                        default="../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final",
                        help="Path to model")
    parser.add_argument("--data_dir", type=str, 
                        default="../SFT/training_data_filtered",
                        help="Directory containing training data")
    parser.add_argument("--data_file", type=str, default="train.jsonl",
                        help="Training data file")
    parser.add_argument("--output_path", type=str, 
                        default="./error_analysis.json",
                        help="Output path for error analysis")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 1: FIND MODEL ERRORS ON TRAINING DATA")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}/{args.data_file}")
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
    
    # Run inference and find errors
    print("\n[3/3] Running inference to find errors...")
    
    errors = []
    correct = []
    results = []
    
    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        expected = get_expected_answer(sample['messages'])
        prompt_text, image_paths = get_prompt_content(sample['messages'])
        
        # Load images
        images = []
        for img_path in image_paths:
            full_path = os.path.join(args.data_dir, img_path)
            if os.path.exists(full_path):
                images.append(Image.open(full_path).convert("RGB"))
        
        # Build conversation for model
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
            }
        ]
        
        # Process input
        inputs = processor.apply_chat_template(
            conversation,
            images=images if images else None,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        
        # Decode response
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        predicted = extract_answer(response)
        
        is_correct = (predicted == expected)
        
        result = {
            "sample_idx": idx,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "prompt_text": prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text,
            "image_paths": image_paths,
            "model_response": response,
        }
        
        results.append(result)
        
        if is_correct:
            correct.append(result)
        else:
            errors.append(result)
    
    # Calculate statistics
    total = len(results)
    num_correct = len(correct)
    num_errors = len(errors)
    accuracy = num_correct / total * 100 if total > 0 else 0
    
    # Count error types
    false_negatives = sum(1 for e in errors if e['expected'] == 'yes' and e['predicted'] == 'no')
    false_positives = sum(1 for e in errors if e['expected'] == 'no' and e['predicted'] == 'yes')
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct: {num_correct} ({accuracy:.2f}%)")
    print(f"Errors: {num_errors} ({100-accuracy:.2f}%)")
    print(f"  - False Negatives (Yes→No): {false_negatives}")
    print(f"  - False Positives (No→Yes): {false_positives}")
    
    # Save results
    output = {
        "metadata": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "data_file": args.data_file,
            "total_samples": total,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "correct": num_correct,
            "errors": num_errors,
            "accuracy": accuracy,
            "false_negatives": false_negatives,
            "false_positives": false_positives
        },
        "errors": errors,
        "all_results": results
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_path}")
    print(f"   Found {num_errors} errors to correct")
    
    # Also save just the error indices for easy reference
    error_indices_path = args.output_path.replace('.json', '_indices.json')
    with open(error_indices_path, 'w') as f:
        json.dump([e['sample_idx'] for e in errors], f)
    
    print(f"   Error indices saved to {error_indices_path}")


if __name__ == "__main__":
    main()

