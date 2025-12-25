# Data Corruption: Generating Negative Instances

## Overview

This module generates **corrupted negative instances** to balance the training data. The original dataset is heavily imbalanced toward positive (Yes) samples.

## Strategy: Trajectory Corruption

```
Original Trajectory (all positive):
A → B → C → D → E
 ✓   ✓   ✓   ✓   ✓

After corruption at step C:
A → B → C' (corrupted)
         ✗ (negative)
```

We take positive samples and corrupt the AFTER image to create realistic negative examples.

## Corruption Types

| Type | Description | Example |
|------|-------------|---------|
| `cross_task` | AFTER from completely different task | Gmail compose → Camera app |
| `wrong_step_future` | AFTER from a later step in same task | Step 3 result → Step 7 result |
| `wrong_step_past` | AFTER from an earlier step in same task | Step 5 result → Step 2 result |
| `no_change` | AFTER = BEFORE (action had no effect) | Same screen before/after |

## Usage

```bash
cd /home/ubuntu/verifier_training/Data_Corruption

# Generate balanced dataset (50% positive, 50% negative)
python3 generate_corruptions.py \
    --data_path ../SFT/training_data_filtered/train.jsonl \
    --output_dir ./corrupted_data \
    --target_ratio 0.5

# Custom ratio (e.g., 40% negative)
python3 generate_corruptions.py --target_ratio 0.4
```

## Output

```
corrupted_data/
├── train_balanced.jsonl    # Balanced training data
└── corruption_stats.json   # Statistics about corruption
```

## Expected Results

```
Original: 647 samples
  - Positive (Yes): ~517 (80%)
  - Negative (No): ~130 (20%)

After corruption: ~1034 samples
  - Positive (Yes): ~517 (50%)
  - Negative (No): ~517 (50%)
```

## Why This Matters

| Imbalanced Data | Balanced Data |
|-----------------|---------------|
| Model biased toward "Yes" | Model learns both cases equally |
| High false positive rate | Better precision/recall balance |
| Poor at detecting failures | Robust failure detection |

## Next Step

After generating balanced data, retrain the model:

```bash
cd ../SFT
python3 train_verifier_4b.py \
    --data_path ../Data_Corruption/corrupted_data/train_balanced.jsonl
```

