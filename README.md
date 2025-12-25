# Qwen3-VL Verifier Training

Train a vision-language model to verify mobile GUI state transitions.

## Training Pipeline

| Step | Folder | Description | Status |
|------|--------|-------------|--------|
| **1** | `SFT/` | Supervised Fine-Tuning on trajectory data | ✅ Done |
| **2** | `Teacher_Student_Distill/` | GPT-5.2 → Qwen distillation | ✅ Done |
| **3+4** | `DPO/` | Data corruption + DPO training | 📝 Ready |

## Project Structure

```
verifier_training/
├── README.md                       # This file
├── setup_environment.sh            # ⭐ Environment setup with patches
├── requirements_train.txt          # Dependencies
│
├── SFT/                            # Step 1: Supervised Fine-Tuning
│   ├── train_verifier.py           # Training script
│   ├── test_verifier.py            # Test any model
│   ├── test_base_model.py          # Test base model
│   ├── training_data_filtered/     # Training data
│   ├── qwen3_vl_verifier/          # Trained 4B model
│   └── qwen3_vl_verifier_8b/       # Trained 8B model
│
├── Teacher_Student_Distill/        # Step 2: Distillation
│   ├── generate_gpt_data.py        # Generate GPT-5.2 responses
│   ├── train_distilled.py          # Train on GPT data
│   ├── gpt_data/                   # GPT responses
│   ├── qwen3_vl_verifier_distilled_4b/   # Distilled 4B
│   └── qwen3_vl_verifier_distilled_8b/   # Distilled 8B
│
└── DPO/                            # Step 3+4: Preference Learning
    ├── generate_dpo_pairs.py       # Generate corruption pairs
    ├── train_dpo.py                # DPO training
    └── README.md                   # Instructions
```

## ⭐ First Time Setup

**Important:** This project requires patches to Unsloth for VLM training.

```bash
# Clone the repo
git clone <your-repo-url>
cd verifier_training

# Run setup script (installs deps + applies patches)
chmod +x setup_environment.sh
./setup_environment.sh
```

The setup script:
1. Installs Python dependencies
2. Applies 3 patches to Unsloth for VLM support
3. Clears compiled cache

## Quick Start

### Step 1: SFT

```bash
cd SFT

# Train 4B model
python3 train_verifier.py --epochs 3

# Train 8B model
python3 train_verifier.py --model unsloth/Qwen3-VL-8B-Instruct --output_dir ./qwen3_vl_verifier_8b

# Test any model
python3 test_verifier.py --model_path ./qwen3_vl_verifier/final --num_samples 71
```

### Step 2: Teacher-Student Distillation

```bash
cd Teacher_Student_Distill

# Set API key
export OPENAI_API_KEY="sk-your-key"

# Generate GPT responses (uses corrected prompt)
python3 generate_gpt_data.py --num_samples all

# Train distilled model
python3 train_distilled.py --base_model ../SFT/qwen3_vl_verifier/final
```

### Step 3+4: DPO Training

```bash
cd DPO

# Generate corruption pairs (~1000 pairs)
python3 generate_dpo_pairs.py

# Train with DPO
python3 train_dpo.py --base_model ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final
```

## Testing Models

Use the same test script for any model:

```bash
cd SFT

# Test SFT model
python3 test_verifier.py --model_path ./qwen3_vl_verifier/final

# Test distilled model
python3 test_verifier.py --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final

# Test DPO model
python3 test_verifier.py --model_path ../DPO/qwen3_vl_verifier_dpo_4b/final
```

## Metrics

| Metric | Description |
|--------|-------------|
| **SR(Traj)** | Success rate at trajectory level (correct predictions) |
| **Tokens (Input)** | Average input tokens per sample |
| **Tokens (Output)** | Average output tokens per sample |

## GPU Requirements

| GPU | VRAM | Step 1 Time | Recommended |
|-----|------|-------------|-------------|
| A100 40GB | 40GB | ~30 min | ⭐ Best |
| A100 80GB | 80GB | ~30 min | ⭐ Best |
| RTX 4090 | 24GB | ~1 hr | ✅ Good |
| RTX 3090 | 24GB | ~1.5 hr | ✅ Good |

## Trained Models

| Model | Location | Description |
|-------|----------|-------------|
| 4B SFT | `SFT/qwen3_vl_verifier/final/` | Base fine-tuned |
| 8B SFT | `SFT/qwen3_vl_verifier_8b/final/` | Base fine-tuned (larger) |
| 4B Distilled | `Teacher_Student_Distill/qwen3_vl_verifier_distilled_4b/final/` | With GPT reasoning |
| 8B Distilled | `Teacher_Student_Distill/qwen3_vl_verifier_distilled_8b/final/` | With GPT reasoning |

## Using a Trained Model

```python
from unsloth import FastModel
from PIL import Image

# Load model
model, processor = FastModel.from_pretrained("./SFT/qwen3_vl_verifier/final")
FastModel.for_inference(model)

# Prepare input
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Your prompt here..."},
    {"type": "image", "image": Image.open("before.jpg")},
    {"type": "image", "image": Image.open("after.jpg")},
]}]

# Generate
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, 
                                        tokenize=True, return_tensors="pt", return_dict=True)
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

## Unsloth Patches

This project requires 3 patches to Unsloth for VLM training:

1. **compiler.py**: Add `import psutil` for CPU detection
2. **rl.py**: Fix VLM detection (remove `_is_vlm = False` override)
3. **dataset_utils.py**: Preserve VLM data collator

These are applied automatically by `setup_environment.sh`.
