# Data Corruption: Generating Negative Instances

Generate corrupted negative samples to create a balanced training dataset.

## Overview

The original dataset is heavily imbalanced (~80% positive, ~20% negative). This module generates realistic negative examples using trajectory-aware corruption to create a balanced 50/50 dataset.

## Corruption Strategy

```
Original Trajectory (all positive):
A → B → C → D → E
 ✓   ✓   ✓   ✓   ✓

After corruption at step C:
A → B → C' (corrupted AFTER image)
         ✗ (negative sample)
```

We take positive samples and replace the AFTER image with an incorrect image to create realistic failure examples.

## Corruption Types

| Type | Probability | Description | Example |
|------|-------------|-------------|---------|
| `cross_task` | 25% | AFTER from completely different task | Gmail → Camera app |
| `wrong_step_future` | 25% | AFTER from a later step in same task | Step 3 → Step 7 |
| `wrong_step_past` | 25% | AFTER from an earlier step in same task | Step 5 → Step 2 |
| `no_change` | 25% | AFTER = BEFORE (action had no effect) | Same screen |

## Usage

```bash
cd Data_Corruption

# Generate balanced dataset (default 50% negative)
python3 generate_corruptions.py

# Custom negative ratio
python3 generate_corruptions.py --target_ratio 0.4  # 40% negative

# Custom input/output
python3 generate_corruptions.py \
    --data_path ../SFT/training_data_filtered/train.jsonl \
    --output_dir ./corrupted_data \
    --target_ratio 0.5
```

## Output

```
corrupted_data/
├── train_balanced.jsonl    # Balanced training data
└── corruption_stats.json   # Statistics
```

## Statistics

After running with default settings:

| Metric | Value |
|--------|-------|
| Original samples | 648 |
| Original positive | 517 (80%) |
| Original negative | 131 (20%) |
| Corruptions added | 387 |
| **Final total** | **1,035** |
| **Final negative ratio** | **50%** |

Corruption breakdown:
- cross_task: ~97 (25%)
- wrong_step_future: ~97 (25%)
- wrong_step_past: ~97 (25%)
- no_change: ~96 (25%)

## Why Balance Matters

| Imbalanced Data | Balanced Data |
|-----------------|---------------|
| Model biased toward "Yes" | Model learns both cases equally |
| High false positive rate | Better precision/recall balance |
| Poor at detecting failures | Robust failure detection |

## How It Works

1. **Load original data** - Read training samples
2. **Count distribution** - Identify positive/negative ratio
3. **Calculate needed corruptions** - Determine how many negatives to generate
4. **Apply corruptions** - For each needed sample:
   - Select random positive sample
   - Choose corruption type (equal probability)
   - Replace AFTER image with corrupted version
   - Generate rejection reason
   - Label as "No"
5. **Combine and shuffle** - Merge original + corrupted samples

## Image Handling

**Important:** This script modifies image *paths* in the JSONL data, not the actual image files. The corrupted samples reference existing images from different samples/tasks.

## Next Steps

After generating balanced data, you can:

1. **Retrain from scratch** on balanced data:
```bash
cd ../SFT
python3 train_verifier.py \
    --data_dir ../Data_Corruption/corrupted_data \
    --output_dir ./qwen_balanced
```

2. **Continue training** existing model on balanced data (fine-tune)

3. **Evaluate impact** of balanced training on false positive rate
