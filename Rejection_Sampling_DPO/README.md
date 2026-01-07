# Rejection Sampling + DPO

## Overview

This approach creates **proper DPO pairs** by:
1. Generating multiple responses from the model for each sample
2. Checking which responses get the correct vs wrong answer
3. Pairing: (correct answer → chosen) vs (wrong answer → rejected)
4. Training with DPO

Unlike the old DPO approach (which corrupted images = different inputs), this keeps the **same input** and only differs in the **output**.

## Why This Works

```
Old DPO (broken):
  Input A + Image X → "Yes"  (chosen)
  Input A + Image Y → "No"   (rejected)  ← Different input!

New DPO (correct):
  Input A + Image X → "Yes, because..."  (chosen)
  Input A + Image X → "No, because..."   (rejected)  ← Same input!
```

## Pipeline

```
Step 1: Generate Multiple Responses
├── For each training sample
├── Generate N responses (with sampling, temp=0.8)
├── Check which are correct vs wrong
└── Output: candidate_responses.jsonl

Step 2: Create DPO Pairs
├── For each sample with both correct & wrong responses
├── Pair them: (correct → chosen, wrong → rejected)
└── Output: dpo_pairs.jsonl

Step 3: Train with DPO
├── Load distilled model
├── Train on preference pairs (manual DPO loss)
└── Output: DPO-trained model
```

## Analysis of 228 DPO-Usable Samples

From 647 training samples with 5 responses each, **228 samples** had both correct and wrong responses (usable for DPO).

### 1. Distribution by Expected Answer

| Expected | Count | % | Error Type When Wrong |
|----------|-------|---|----------------------|
| YES | 162 | 71.1% | False Negative (model too conservative) |
| NO | 66 | 28.9% | False Positive (model too lenient) |

**Key Insight**: The model makes more errors on YES cases, meaning it tends to be **too conservative** (saying No when it should say Yes).

### 2. Uncertainty Distribution

| Model Accuracy | Samples | Interpretation |
|----------------|---------|----------------|
| 4/5 correct | 85 | Low uncertainty (1 error) |
| 3/5 correct | 59 | Medium uncertainty |
| 2/5 correct | 51 | High uncertainty |
| 1/5 correct | 33 | Very uncertain (mostly wrong) |

### 3. DPO Pair Options

| Strategy | Pairs | Notes |
|----------|-------|-------|
| All wrong responses | 488 | Maximum training signal |
| Max 2 per sample | 371 | Good balance |
| 1 per sample | 228 | Conservative |

**We used 488 pairs** (all wrong responses paired with correct ones) for maximum training signal.

### 4. Non-Usable Samples

| Category | Count | Meaning |
|----------|-------|---------|
| All 5 Correct | 375 | Model already handles these well |
| All 5 Wrong | 44 | Model consistently fails (may need different approach) |

## Usage

### Step 1: Generate Candidate Responses

```bash
python3 generate_candidates.py \
    --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
    --data_dir ../SFT/training_data_filtered \
    --num_responses 5 \
    --temperature 0.8 \
    --output_path ./candidate_responses.jsonl
```

### Step 2: Create DPO Pairs

```bash
python3 create_dpo_pairs.py \
    --candidates_path ./candidate_responses.jsonl \
    --output_path ./dpo_pairs_488.jsonl
```

### Step 3: Train with DPO

**Note**: Use `train_dpo_simple.py` (not the TRL-based scripts which don't work with VLMs).

```bash
python3 train_dpo_simple.py \
    --pairs_path ./dpo_pairs_488.jsonl \
    --output_dir ./qwen3_vl_verifier_dpo_8b_v2
```

### Step 4: Test the Model

```bash
cd ../SFT && python3 test_verifier.py \
    --model_path ../Rejection_Sampling_DPO/qwen3_vl_verifier_dpo_8b_v2/final \
    --data_dir ../testing_data_androidworld \
    --val_file test.jsonl \
    --num_samples 1082
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `num_responses` | Responses per sample | 5 |
| `temperature` | Sampling temperature | 0.8 |
| `beta` | DPO preference strength | 0.1 |
| `grad_accum` | Gradient accumulation | 16 |
| `lr` | Learning rate | 5e-6 |

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Distilled Qwen3-8B (before DPO) | 79.21% | Baseline |
| DPO-trained (488 pairs) | TBD | After DPO |

## Technical Notes

- **TRL DPOTrainer doesn't work with VLMs** - it gets stuck during tokenization
- We use `train_dpo_simple.py` which implements DPO loss manually
- Images are resized to 512x512 to reduce VLM token count
- `torch.autocast(dtype=bfloat16)` is required for stable training
