# Q-Verifier: Vision-Language Model Verifiers for GUI Action Validation

Train vision-language models to verify whether GUI actions successfully contribute to task completion.

## 🎯 Best Result: 82.35% Accuracy

Achieved with **Distillation → DPO → GRPO** stacking on Qwen3-VL-8B, surpassing GPT-4o (66.17%) and GPT-5.2 (70.33%) by over 12 percentage points.

## Overview

This project develops VLM verifiers that take a BEFORE screenshot, AFTER screenshot, action description, and subgoal, then determine whether the action made valid progress. The verifier outputs a binary verdict (Yes/No) with supporting reasoning.

## Training Pipeline

| Step | Folder | Description | Best Accuracy |
|------|--------|-------------|---------------|
| **1** | `SFT/` | Supervised Fine-Tuning | 81.52% |
| **2** | `Teacher_Student_Distill/` | GPT-5.2 → Qwen distillation | 79.21% |
| **3** | `Data_Corruption/` | Balanced dataset with negatives | 77.08% |
| **4** | `Rejection_Sampling_DPO/` | Direct Preference Optimization | 82.07% |
| **5** | `GRPO/` | Group Relative Policy Optimization | 82.16% |
| **6** | `GRPO/` | DPO → GRPO Stacking | **82.35%** |

## Results Summary

| Method | Qwen3-VL-8B | Δ vs Base |
|--------|-------------|-----------|
| Base (no fine-tuning) | 73.11% | - |
| GPT-4o | 66.17% | -6.94% |
| GPT-5.2 | 70.33% | -2.78% |
| SFT | 81.52% | +8.41% |
| Distilled | 79.21% | +6.10% |
| DPO (on Distilled) | 82.07% | +8.96% |
| GRPO (on Distilled) | 82.16% | +9.05% |
| **DPO → GRPO** | **82.35%** | **+9.24%** |

## Project Structure

```
Q-Verifier/
├── README.md                       # This file
├── RESEARCH_PAPER.tex              # Research paper (LaTeX)
├── RESEARCH_PAPER.pdf              # Compiled paper
├── setup_environment.sh            # Environment setup with patches
├── .gitignore                      # Excludes models & images
│
├── SFT/                            # Step 1: Supervised Fine-Tuning
│   ├── train_verifier.py           # Training script
│   ├── test_verifier.py            # Evaluation script
│   ├── test_base_model.py          # Test base models
│   ├── test_gpt.py                 # Test GPT models
│   └── training_data_filtered/     # Training data (647 samples)
│
├── Teacher_Student_Distill/        # Step 2: Distillation
│   ├── generate_gpt_data.py        # Generate GPT-5.2 responses
│   └── train_distilled.py          # Train on GPT data
│
├── Data_Corruption/                # Step 3: Negative Sample Generation
│   ├── generate_corruptions.py     # Trajectory-aware corruption
│   └── README.md                   # Corruption strategy details
│
├── Rejection_Sampling_DPO/         # Step 4: DPO Training
│   ├── generate_candidates.py      # Generate candidate responses
│   ├── build_dpo_pairs.py          # Build preference pairs
│   ├── train_dpo_simple.py         # Custom DPO training for VLMs
│   └── dpo_pairs_488.jsonl         # 488 preference pairs
│
├── GRPO/                           # Step 5: GRPO Training
│   ├── train_grpo.py               # GRPO with smart reward function
│   └── README.md                   # GRPO details & reward design
│
├── Results/                        # All evaluation results
│   ├── Base_Results/               # Base model results
│   ├── SFT_Results/                # SFT results
│   ├── Distilled_Results/          # Distillation results
│   ├── Corruption_SFT/             # Balanced data results
│   ├── DPO_res/                    # DPO results
│   └── GRPO/                       # GRPO results
│
└── testing_data_androidworld/      # Test set (1,082 samples)
```

## First Time Setup

### Option 1: From GitHub (code only)
```bash
# Clone the repo
git clone https://github.com/pockyitachi/Q-Verifier.git
cd Q-Verifier

# Run setup script (installs deps + applies patches)
chmod +x setup_environment.sh
./setup_environment.sh

# Download and extract models (if needed)
# Get Q-Verifier-models.zip from shared storage
unzip Q-Verifier-models.zip -d .
```

### Option 2: From Zip Files
```bash
# Extract code
unzip Q-Verifier-code.zip
cd Q-Verifier

# Run setup
chmod +x setup_environment.sh
./setup_environment.sh

# Extract models (unzip in project root - paths are preserved)
unzip Q-Verifier-models.zip -d .
```

The setup script:
1. Installs Python dependencies from `requirements.txt`
2. Applies patches to Unsloth for VLM support
3. Verifies installation

## Model Locations

After extracting `Q-Verifier-models.zip`, models will be in these locations:

```
Q-Verifier/
├── SFT/
│   ├── qwen3_vl_verifier_4b/final/          # SFT Qwen3-4B
│   ├── qwen3_vl_verifier_8b/final/          # SFT Qwen3-8B
│   ├── qwen25_vl_verifier_3b/final/         # SFT Qwen2.5-3B
│   ├── qwen25_vl_verifier_7b/final/         # SFT Qwen2.5-7B
│   ├── qwen3_vl_verifier_balanced_4b/final/ # Balanced SFT 4B
│   ├── qwen3_vl_verifier_balanced_8b/final/ # Balanced SFT 8B
│   ├── qwen25_vl_verifier_balanced_3b/final/
│   └── qwen25_vl_verifier_balanced_7b/final/
│
├── Teacher_Student_Distill/
│   ├── qwen3_vl_verifier_distilled_4b/final/  # Distilled 4B
│   ├── qwen3_vl_verifier_distilled_8b/final/  # Distilled 8B (base for DPO/GRPO)
│   ├── qwen25_vl_verifier_distilled_3b/final/
│   └── qwen25_vl_verifier_distilled_7b/final/
│
├── Rejection_Sampling_DPO/
│   ├── qwen3_vl_distilled_dpo_8b_488/final/   # DPO on Distilled (82.07%)
│   ├── qwen3_vl_sft_dpo_8b_488/final/         # DPO on SFT
│   └── qwen3_vl_grpo_dpo_8b_stacked/final/    # GRPO→DPO stacked
│
└── GRPO/
    ├── qwen3_vl_distilled_grpo_8b_1e_6/final/ # GRPO on Distilled (82.16%)
    ├── qwen3_vl_distilled_dpo_grpo_8b/final/  # DPO→GRPO stacked (82.35%) ⭐ BEST
    ├── qwen3_vl_distilled_dpo_grpo_4b/final/  # DPO→GRPO 4B
    └── qwen3_vl_sft_grpo_8b_1e_6/final/       # GRPO on SFT
```

## Quick Start

### Step 1: SFT Training

```bash
cd SFT

# Train Qwen3-VL-8B (best base for optimization)
python3 train_verifier.py --model unsloth/Qwen3-VL-8B-Instruct --output_dir ./qwen3_vl_verifier_8b
```

### Step 2: Teacher-Student Distillation

```bash
cd Teacher_Student_Distill
export OPENAI_API_KEY="sk-your-key"

# Generate GPT-5.2 responses
python3 generate_gpt_data.py --num_samples all

# Train distilled model (required for DPO/GRPO)
python3 train_distilled.py --base_model ../SFT/qwen3_vl_verifier_8b/final \
                           --output_dir ./qwen3_vl_verifier_distilled_8b
```

### Step 3: DPO Training

```bash
cd Rejection_Sampling_DPO

# Generate candidate responses (5 per sample)
python3 generate_candidates.py

# Build preference pairs
python3 build_dpo_pairs.py

# Train DPO (17 min, +2.86% accuracy)
python3 train_dpo_simple.py --pairs_path dpo_pairs_488.jsonl --epochs 1 --beta 0.1
```

### Step 4: GRPO Training

```bash
cd GRPO

# Train GRPO on distilled model (4h, +2.95% accuracy)
python3 train_grpo.py --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final \
                      --learning_rate 1e-6 --epochs 1
```

### Testing Models

```bash
cd SFT

# Test any model
python3 test_verifier.py --model_path ../GRPO/qwen3_vl_distilled_dpo_grpo_8b/final \
                         --data_dir ../testing_data_androidworld \
                         --val_file test.jsonl \
                         --num_samples 1082
```

## Key Findings

1. **Fine-tuning dramatically improves accuracy**: SFT achieves 81.52%, +8.41% over base
2. **Distillation enables RL optimization**: Structured reasoning from distillation provides diverse training signal for DPO/GRPO
3. **DPO and GRPO provide comparable gains**: Both improve ~3%, but DPO is faster (17min vs 4h)
4. **Stacking works**: DPO → GRPO achieves best result (82.35%)
5. **Reasoning style matters**: GRPO improves distilled models (+2.95%) but degrades SFT models (-2.78%)

## Supported Models

| Model Family | Variants | Tested |
|--------------|----------|--------|
| Qwen3-VL | 4B, 8B | ✅ |
| Qwen2.5-VL | 3B, 7B | ✅ |
| GPT-4o | - | ✅ |
| GPT-5.2 | - | ✅ |

## GPU Requirements

| GPU | VRAM | Recommended For |
|-----|------|-----------------|
| A100 40GB+ | 40GB+ | 7B/8B models, GRPO ⭐ |
| RTX 4090 | 24GB | 3B/4B models |

## Citation

If you use this code, please cite:
```
@misc{qverifier2025,
  title={Q-Verifier: Training Vision-Language Models as GUI Action Verifiers},
  year={2025}
}
```

## License

MIT
