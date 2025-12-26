# DPO: Direct Preference Optimization (Experimental)

⚠️ **Status: Experimental** - DPO with VLM images has technical challenges.

## Overview

Direct Preference Optimization (DPO) trains the model to prefer correct verifications over incorrect ones using preference pairs.

## Pipeline

```
Step 1: Generate DPO Pairs
├── Take positive (YES) samples
├── Create corrupted versions (swap AFTER images)
├── Pair: (correct → chosen) vs (corrupted → rejected)
└── Output: dpo_pairs/dpo_pairs.jsonl

Step 2: DPO Training (Experimental)
├── Load distilled model
├── Train with preference pairs
└── Output: trained DPO model
```

## Corruption Types

| Type | Description | Percentage |
|------|-------------|------------|
| `cross_task` | AFTER from different task | ~50% |
| `wrong_step` | AFTER from different step in same task | ~30% |
| `no_change` | AFTER = BEFORE | ~20% |

## Generated Data

DPO pairs have been generated and are available:

```
dpo_pairs/
├── dpo_pairs.jsonl    # 1,035 preference pairs
└── summary.json       # Generation statistics
```

## Usage

### Generate DPO Pairs

```bash
cd DPO
python3 generate_dpo_pairs.py \
    --data_path ../SFT/training_data_filtered/train.jsonl \
    --output_dir ./dpo_pairs \
    --corruptions_per_sample 2
```

### DPO Training (Experimental)

```bash
# Note: May encounter issues with VLM image processing
python3 train_dpo.py \
    --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final \
    --pairs_path ./dpo_pairs/dpo_pairs.jsonl \
    --output_dir ./qwen3_vl_verifier_dpo_4b
```

## Known Issues

DPO training with VLM images faces challenges:

1. **Image serialization** - PIL images don't serialize well for multiprocessing
2. **Memory usage** - Processing both chosen/rejected images doubles memory
3. **Slow training** - Image processing significantly slows training

## Alternative Approaches

If DPO doesn't work well, consider:

1. **Balanced SFT** - Train on balanced dataset from `Data_Corruption/`
2. **Text-only DPO** - DPO without images (loses visual context)
3. **Rejection sampling** - Generate multiple responses, filter by quality

## Why DPO?

| Method | Learns |
|--------|--------|
| SFT | "Output X for input Y" |
| DPO | "Prefer X over Z for input Y" |

DPO is ideal for verifiers because:
- Teaches relative preferences (correct > incorrect)
- Can be more sample-efficient than adding negatives to SFT
- Aligns model behavior with human preferences

## DPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beta` | 0.1 | Preference strength (higher = stronger) |
| `--lr` | 5e-5 | Learning rate (lower than SFT) |
| `--epochs` | 1 | DPO typically needs fewer epochs |
