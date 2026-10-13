# Self-Improvement: Targeted Error Correction

Original error-directed teacher-correction workflow. The 647-example diagram below describes its original data export, not the maintained v9 training population. This workflow is preserved separately from the later Sol correction rounds.

## Overview

This pipeline improves the model by:
1. Finding samples where the model makes errors (on training data)
2. Using GPT-5.2 to generate correct responses for those errors
3. Fine-tuning the model on these corrections

This is more efficient than full teacher-student distillation because it only focuses on the model's weaknesses.

## Pipeline

```
Step 1: Find Errors
├── Run distilled model on training data (647 samples)
├── Compare predictions to ground truth
├── Identify samples with wrong predictions
└── Output: error_samples.jsonl

Step 2: Get GPT-5.2 Corrections
├── Send error samples to GPT-5.2
├── Get correct responses with reasoning
└── Output: corrections.jsonl

Step 3: Train on Corrections
├── Load distilled model
├── Fine-tune on correction data
└── Output: improved model
```

## Usage

### Step 1: Find Model Errors on Training Data

```bash
python3 find_errors.py \
    --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
    --data_dir ../SFT/training_data_filtered \
    --output_path ./error_analysis.json
```

### Step 2: Get GPT-5.2 Corrections

```bash
export OPENAI_API_KEY="your-key-here"

python3 get_corrections.py \
    --error_file ./error_analysis.json \
    --data_dir ../SFT/training_data_filtered \
    --output_path ./corrections.jsonl
```

### Step 3: Train Improved Model

```bash
python3 train_improved.py \
    --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
    --corrections_path ./corrections.jsonl \
    --data_dir ../SFT/training_data_filtered \
    --output_dir ./qwen3_vl_verifier_improved_8b
```

## Results

This guide describes the method, not a measured improvement guarantee. Use the
saved evaluator output for the corresponding checkpoint and dataset.

## Original motivation

1. **Targeted learning** - Only fixes what's broken
2. **High-quality corrections** - GPT-5.2 provides expert-level reasoning
3. **No data leakage** - Errors found on training data, tested on AndroidWorld
4. **Efficient** - ~100-150 GPT-5.2 calls instead of 647

