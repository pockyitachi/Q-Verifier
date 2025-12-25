# Step 3+4: Data Corruption & DPO Training

## Overview

This step generates corrupted negative examples and uses Direct Preference Optimization (DPO) to teach the model to **prefer correct transitions over incorrect ones**.

## Pipeline

```
Step 3: Generate DPO Pairs (Corruption)
├── Take YES samples from training data
├── Create corrupted versions (swap AFTER images)
├── Pair: (correct → chosen) vs (corrupted → rejected)
└── Output: dpo_pairs/dpo_pairs.jsonl

Step 4: DPO Training
├── Load distilled model (Step 2 output)
├── Train with preference pairs
└── Output: qwen3_vl_verifier_dpo_4b/
```

## Corruption Types

| Type | Method | Example |
|------|--------|---------|
| **cross_task** | AFTER from different task | Gmail compose → Camera app |
| **wrong_step** | AFTER from different step in same task | Step 3 result → Step 5 result |
| **no_change** | AFTER = BEFORE | Action had no effect |

## Usage

### Step 3: Generate DPO Pairs

```bash
cd DPO
python generate_dpo_pairs.py \
    --data_path ../SFT/training_data_filtered/train.jsonl \
    --output_dir ./dpo_pairs \
    --corruptions_per_sample 2
```

Output:
- `dpo_pairs/dpo_pairs.jsonl` - Preference pairs
- `dpo_pairs/summary.json` - Generation summary

### Step 4: DPO Training

```bash
# 4B model
python train_dpo.py \
    --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final \
    --pairs_path ./dpo_pairs/dpo_pairs.jsonl \
    --output_dir ./qwen3_vl_verifier_dpo_4b

# 8B model
python train_dpo.py \
    --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
    --pairs_path ./dpo_pairs/dpo_pairs.jsonl \
    --output_dir ./qwen3_vl_verifier_dpo_8b
```

## Expected Output

```
DPO PAIR GENERATION
============================================================
Data: ../SFT/training_data_filtered/train.jsonl
Output: ./dpo_pairs
Corruptions per sample: 2

[1/3] Loading training data...
Loaded 647 samples

[2/3] Generating DPO pairs...
Found 517 YES samples for pair generation
Generated ~1000 preference pairs

Corruption type distribution:
  cross_task: 517 (50%)
  wrong_step: ~300 (30%)
  no_change: ~200 (20%)
```

## DPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beta` | 0.1 | Controls preference strength (higher = stronger preference) |
| `--lr` | 5e-5 | Learning rate (lower than SFT) |
| `--epochs` | 1 | DPO typically needs fewer epochs |

## Why DPO?

| Method | Learns |
|--------|--------|
| SFT | "Output X for input Y" |
| DPO | "Prefer X over Z for input Y" |

DPO is ideal for verifiers because:
- Teaches relative preferences (correct > incorrect)
- More sample-efficient than adding negatives to SFT
- Aligns model behavior with human preferences

