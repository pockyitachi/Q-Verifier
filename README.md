# Q-Verifier: Vision-Language Model Verifiers for GUI Action Validation

Train vision-language models to verify whether GUI actions successfully contribute to task completion.

## Overview

This project develops VLM verifiers that take a BEFORE screenshot, AFTER screenshot, action description, and subgoal, then determine whether the action made valid progress. The verifier outputs a binary verdict (Yes/No) with supporting reasoning.

## Training Pipeline

| Step | Folder | Description | Status |
|------|--------|-------------|--------|
| **1** | `SFT/` | Supervised Fine-Tuning on trajectory data | ✅ Done |
| **2** | `Teacher_Student_Distill/` | GPT-4 → Qwen distillation | ✅ Done |
| **3** | `Data_Corruption/` | Generate balanced dataset with negatives | ✅ Done |
| **4** | `DPO/` | Direct Preference Optimization | 🔬 Experimental |

## Trained Models

### Qwen3-VL Models
| Model | Type | Location |
|-------|------|----------|
| Qwen3-VL-4B | SFT | `SFT/qwen3_vl_verifier_4b/final/` |
| Qwen3-VL-8B | SFT | `SFT/qwen3_vl_verifier_8b/final/` |
| Qwen3-VL-4B | Distilled | `Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final/` |
| Qwen3-VL-8B | Distilled | `Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final/` |

### Qwen2.5-VL Models
| Model | Type | Location |
|-------|------|----------|
| Qwen2.5-VL-3B | SFT | `SFT/qwen25_vl_verifier_3b/final/` |
| Qwen2.5-VL-7B | SFT | `SFT/qwen25_vl_verifier_7b/final/` |
| Qwen2.5-VL-3B | Distilled | `Teacher_Student_Distill/qwen25_vl_verifier_distilled_3b/final/` |
| Qwen2.5-VL-7B | Distilled | `Teacher_Student_Distill/qwen25_vl_verifier_distilled_7b/final/` |

## Project Structure

```
verifier_training/
├── README.md                       # This file
├── RESEARCH_DOCUMENTATION.txt      # Detailed research documentation
├── setup_environment.sh            # ⭐ Environment setup with patches
│
├── SFT/                            # Step 1: Supervised Fine-Tuning
│   ├── train_verifier.py           # Training script (supports Qwen3 & Qwen2.5)
│   ├── test_verifier.py            # Evaluation script
│   ├── training_data_filtered/     # Training data with images
│   ├── qwen3_vl_verifier_4b/       # Qwen3-VL 4B model
│   ├── qwen3_vl_verifier_8b/       # Qwen3-VL 8B model
│   ├── qwen25_vl_verifier_3b/      # Qwen2.5-VL 3B model
│   └── qwen25_vl_verifier_7b/      # Qwen2.5-VL 7B model
│
├── Teacher_Student_Distill/        # Step 2: Distillation
│   ├── generate_gpt_data.py        # Generate GPT-4 responses
│   ├── train_distilled.py          # Train on GPT data
│   ├── gpt_data/                   # GPT responses
│   ├── qwen3_vl_verifier_distilled_4b/
│   ├── qwen3_vl_verifier_distilled_8b/
│   ├── qwen25_vl_verifier_distilled_3b/
│   └── qwen25_vl_verifier_distilled_7b/
│
├── Data_Corruption/                # Step 3: Negative Sample Generation
│   ├── generate_corruptions.py     # Trajectory-aware corruption
│   └── corrupted_data/             # Balanced dataset output
│
└── DPO/                            # Step 4: Preference Learning (Experimental)
    ├── generate_dpo_pairs.py       # Generate preference pairs
    └── train_dpo.py                # DPO training
```

## First Time Setup

**Important:** This project requires patches to Unsloth for VLM training.

```bash
# Clone the repo
git clone https://github.com/pockyitachi/Q-Verifier.git
cd Q-Verifier

# Run setup script (installs deps + applies patches)
chmod +x setup_environment.sh
./setup_environment.sh
```

The setup script:
1. Installs Python dependencies (torch, unsloth, trl, transformers, etc.)
2. Applies 3 patches to Unsloth for VLM support
3. Clears compiled cache

## Quick Start

### Step 1: SFT Training

```bash
cd SFT

# Train Qwen3-VL-4B
python3 train_verifier.py --model unsloth/Qwen3-VL-4B-Instruct --output_dir ./qwen3_vl_verifier_4b

# Train Qwen2.5-VL-3B
python3 train_verifier.py --model unsloth/Qwen2.5-VL-3B-Instruct --output_dir ./qwen25_vl_verifier_3b

# Train Qwen2.5-VL-7B
python3 train_verifier.py --model unsloth/Qwen2.5-VL-7B-Instruct --output_dir ./qwen25_vl_verifier_7b
```

### Step 2: Teacher-Student Distillation

```bash
cd Teacher_Student_Distill

# Set API key
export OPENAI_API_KEY="sk-your-key"

# Generate GPT-4 responses
python3 generate_gpt_data.py --num_samples all

# Train distilled model
python3 train_distilled.py --base_model ../SFT/qwen25_vl_verifier_3b/final \
                           --output_dir ./qwen25_vl_verifier_distilled_3b
```

### Step 3: Data Corruption (Balanced Dataset)

```bash
cd Data_Corruption

# Generate balanced dataset (50% positive, 50% negative)
python3 generate_corruptions.py --target_ratio 0.5
```

### Testing Models

```bash
cd SFT

# Test any trained model
python3 test_verifier.py --model_path ./qwen25_vl_verifier_3b/final --num_samples 71
python3 test_verifier.py --model_path ../Teacher_Student_Distill/qwen25_vl_verifier_distilled_3b/final
```

## Model Usage

```python
from unsloth import FastModel
from PIL import Image

# Load model
model, processor = FastModel.from_pretrained("./SFT/qwen25_vl_verifier_3b/final")
model.for_inference()

# Prepare input
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Your verification prompt..."},
    {"type": "image", "image": Image.open("before.jpg")},
    {"type": "image", "image": Image.open("after.jpg")},
]}]

# Generate
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, 
                                        tokenize=True, return_tensors="pt", return_dict=True)
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

## Supported Models

| Model Family | Variants | Status |
|--------------|----------|--------|
| Qwen3-VL | 4B, 8B | ✅ Trained |
| Qwen2.5-VL | 3B, 7B | ✅ Trained |

## Metrics

| Metric | Description |
|--------|-------------|
| **SR(Traj)** | Success rate on trajectory validation (correct predictions) |
| **Tokens (Input)** | Average input tokens per sample |
| **Tokens (Output)** | Average output tokens per sample |

## GPU Requirements

| GPU | VRAM | Recommended For |
|-----|------|-----------------|
| A100 40GB+ | 40GB+ | 7B/8B models ⭐ |
| RTX 4090 | 24GB | 3B/4B models |
| RTX 3090 | 24GB | 3B/4B models |

## Documentation

See `RESEARCH_DOCUMENTATION.txt` for detailed methodology, implementation details, and experimental design.

## Repository

GitHub: https://github.com/pockyitachi/Q-Verifier.git
