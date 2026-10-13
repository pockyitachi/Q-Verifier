# Teacher-Student Distillation

Original paper teacher-response distillation. Dataset sizes and two-field examples below describe the original exports; the maintained data/prompt contracts are indexed in [datasets](../../datasets/README.md). The separate v9 pipeline supervises all four response fields.

Generate teacher responses, then continue training the Qwen student on them. The current CLI default in this original script is `gpt-5.2`; the teacher is configurable.

## Overview

The original workflow trains on teacher-generated structured responses:

```
SFT Model → GPT-4 Generates Better Responses → Train on GPT Responses → Distilled Model
```

Classification accuracy and explanation quality must be checked on each resulting checkpoint; the method description alone does not establish an improvement.

## Files

| File | Description |
|------|-------------|
| `generate_gpt_data.py` | Generate GPT-4 responses for training data |
| `train_distilled.py` | Train SFT model on GPT responses |

## Trained Models

| Model | Base Model | Directory |
|-------|------------|-----------|
| Qwen3-VL-4B Distilled | qwen3_vl_verifier_4b | `qwen3_vl_verifier_distilled_4b/final/` |
| Qwen3-VL-8B Distilled | qwen3_vl_verifier_8b | `qwen3_vl_verifier_distilled_8b/final/` |
| Qwen2.5-VL-3B Distilled | qwen25_vl_verifier_3b | `qwen25_vl_verifier_distilled_3b/final/` |
| Qwen2.5-VL-7B Distilled | qwen25_vl_verifier_7b | `qwen25_vl_verifier_distilled_7b/final/` |

## Usage

### Step 1: Generate Teacher Responses

```bash
cd training/Teacher_Student_Distill

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Generate responses (test with small sample first)
python3 generate_gpt_data.py --num_samples 10

# Original 647-row export; replace 647 with the intended input count
python3 generate_gpt_data.py --num_samples 647

# Resume if interrupted
python3 generate_gpt_data.py --num_samples 647 --start_idx 500
```

### Step 2: Train Distilled Model

```bash
# Qwen3-VL-4B Distilled
python3 train_distilled.py \
    --base_model ../SFT/qwen3_vl_verifier_4b/final \
    --output_dir ./qwen3_vl_verifier_distilled_4b

# Qwen2.5-VL-3B Distilled
python3 train_distilled.py \
    --base_model ../SFT/qwen25_vl_verifier_3b/final \
    --output_dir ./qwen25_vl_verifier_distilled_3b

# Qwen2.5-VL-7B Distilled
python3 train_distilled.py \
    --base_model ../SFT/qwen25_vl_verifier_7b/final \
    --output_dir ./qwen25_vl_verifier_distilled_7b
```

### Step 3: Test Distilled Model

```bash
cd ../SFT
python3 test_verifier.py --model_path ../Teacher_Student_Distill/qwen25_vl_verifier_distilled_3b/final
```

## Original Response Format

The original export used the following two-field response structure:

```json
{
  "answer": "Yes",
  "reason": "1) BEFORE: [description of initial state]
             2) AFTER: [description of final state]
             3) CHANGE: [what changed between states]
             4) EVALUATION: [whether change aligns with subgoal]
             5) FINAL: [verdict with explanation]"
}
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate (lower than SFT) |
| `--max_seq_length` | 8192 | Max sequence length (longer for detailed responses) |

## GPT Model Options

| Model | Recommended |
|-------|-------------|
| `gpt-5.2` | CLI default |
| `gpt-4o` | Supported alternative |
| `gpt-4o-mini` | Budget option |

## Data

Generated GPT data is stored in `gpt_data/`:
- `gpt_responses_0_647.jsonl` - Raw GPT responses
- `train_distilled.jsonl` - Formatted training data
- `train_distilled_corrected.jsonl` - Corrected training data

## Key Differences: SFT vs Distilled

| Aspect | SFT Model | Distilled Model |
|--------|-----------|-----------------|
| Response length | ~60 tokens | ~180 tokens |
| Reasoning style | Brief justification | Detailed chain-of-thought |
| Training data | Human labels | GPT-4 generated |
