#!/usr/bin/env python3
"""
Step 2: Create DPO Pairs from Candidate Responses

This script:
1. Loads candidate responses from Step 1
2. For samples with both correct and wrong responses
3. Creates DPO pairs (correct → chosen, wrong → rejected)
4. Saves pairs for DPO training

Usage:
    python3 create_dpo_pairs.py \
        --candidates_path ./candidate_responses.jsonl \
        --output_path ./dpo_pairs.jsonl
"""

import argparse
import json
import random
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Create DPO pairs from candidates")
    parser.add_argument("--candidates_path", type=str, 
                        default="./candidate_responses.jsonl",
                        help="Path to candidate responses from Step 1")
    parser.add_argument("--output_path", type=str, 
                        default="./dpo_pairs.jsonl",
                        help="Output path for DPO pairs")
    parser.add_argument("--max_pairs_per_sample", type=int, default=None,
                        help="Max pairs per sample (default: None = all wrong responses)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("STEP 2: CREATE DPO PAIRS")
    print("=" * 60)
    print(f"Candidates: {args.candidates_path}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # Load candidates
    print("\n[1/2] Loading candidate responses...")
    
    candidates = []
    with open(args.candidates_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    
    # Filter to samples with both correct and wrong
    usable = [c for c in candidates if c['has_both']]
    
    print(f"✅ Loaded {len(candidates)} samples")
    print(f"   Usable (has both correct & wrong): {len(usable)}")
    
    # Create pairs
    print("\n[2/2] Creating DPO pairs...")
    
    pairs = []
    
    for candidate in usable:
        correct_responses = candidate['correct_responses']
        wrong_responses = candidate['wrong_responses']
        
        # Limit wrong responses if max_pairs_per_sample is set
        if args.max_pairs_per_sample:
            wrong_to_use = wrong_responses[:args.max_pairs_per_sample]
        else:
            wrong_to_use = wrong_responses
        
        # Create pairs: one for each wrong response
        # Each wrong response gets paired with a correct response
        for i, rejected in enumerate(wrong_to_use):
            # Cycle through correct responses if fewer correct than wrong
            chosen = correct_responses[i % len(correct_responses)]
            
            pair = {
                "sample_idx": candidate['sample_idx'],
                "prompt": candidate['prompt_text'],
                "image_paths": candidate['image_paths'],
                "chosen": chosen,
                "rejected": rejected,
                "expected": candidate['expected']
            }
            pairs.append(pair)
    
    # Save pairs
    with open(args.output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    # Save summary
    summary = {
        "total_candidates": len(candidates),
        "usable_candidates": len(usable),
        "total_pairs": len(pairs),
        "max_pairs_per_sample": args.max_pairs_per_sample or "all",
        "seed": args.seed,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_path = args.output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("DPO PAIRS CREATED")
    print("=" * 60)
    print(f"Total pairs: {len(pairs)}")
    print(f"✅ Saved to {args.output_path}")
    print(f"   Summary: {summary_path}")
    print(f"\nNext step: python3 train_dpo.py --pairs_path {args.output_path}")


if __name__ == "__main__":
    main()

