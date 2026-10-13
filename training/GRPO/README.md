# GRPO: Group Relative Policy Optimization

Original paper GRPO trainer and asymmetric reward. The analysis and settings below belong to that implementation. The newer four-field reward is separately specified in [REWARD_V9.txt](REWARD_V9.txt); its existence does not change the original trainer or establish the scope of a future trainer loss.

## Overview

GRPO is a reinforcement learning algorithm (used by DeepSeek-R1) that:
1. Generates multiple responses per prompt
2. Scores each response with a reward function
3. Learns to prefer high-reward responses

## Error Analysis (Distilled 8B Model)

Before designing the reward function, we analyzed errors from the distilled model:

```
Total errors: 225/1082 (20.8%)
├── False Negatives (Yes→No): 172 (76.4% of errors)  ← MAIN PROBLEM
└── False Positives (No→Yes):  53 (23.6% of errors)
```

### Error Patterns

**False Negatives (Model Too Strict):**
- Model over-analyzes intermediate steps
- Expects final outcome immediately
- Common phrases: "does not directly", "not aligned with", "no visible change"

**False Positives (Model Too Lenient):**
- Model says Yes because "action is a step toward goal"
- But doesn't verify if the action actually SUCCEEDED
- Verifier must check BOTH: (1) right action? AND (2) did it work?

## Smart Reward Function

Based on error analysis, the reward function is **asymmetric**:

| Prediction | Expected | Reward | Reasoning |
|------------|----------|--------|-----------|
| Yes (correct) | Yes | **+2.5** | Encourage correct Yes (reduce FN) |
| No (correct) | No | +2.0 | Standard reward |
| Yes (wrong) | No | -2.0 | Standard penalty |
| No (wrong) | Yes | **-2.5** | Extra penalty (FN is 76% of errors) |

### Additional Adjustments

1. **Pattern Detection**:
   - **FN patterns** (-0.1): "does not directly", "not aligned with", "no visible change"
   - **FP patterns** (-0.2): Says "step toward goal" without checking outcome
   - Good: Mentions "result", "succeeded", "failed", "error", "unchanged"

2. **Reasoning Length**:
   - Good (100-400 chars): +0.1
   - Too short (<50 chars): -0.2
   - Too long (>600 chars): -0.1

## Usage

```bash
cd training/GRPO

python3 train_grpo.py \
    --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
    --data_dir ../SFT/training_data_filtered \
    --output_dir ./qwen3_vl_verifier_grpo_8b \
    --num_generations 4 \
    --epochs 1
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | distilled_8b | Model to start from |
| `--num_generations` | 4 | Responses generated per prompt |
| `--epochs` | 1 | Training epochs |
| `--lr` | 1e-6 | Learning rate |
| `--temperature` | 0.8 | Sampling temperature |

## Original design hypotheses

The asymmetric rewards should:
1. **Reduce False Negatives**: Model learns that saying "No" incorrectly is expensive
2. **Maintain True Negatives**: Correct "No" still gets +2.0 reward
3. **Improve calibration**: Pattern detection discourages over-analysis

## Testing

After training:

```bash
cd ../SFT
python3 test_verifier.py \
    --model_path ../GRPO/qwen3_vl_verifier_grpo_8b/final \
    --data_dir ../testing_data_androidworld \
    --val_file test.jsonl
```

Compare with original distilled model:
- Original: 79.2% accuracy (172 FN, 53 FP)
- Target: 81%+ accuracy with reduced FN
