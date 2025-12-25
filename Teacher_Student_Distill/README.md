# Teacher-Student Distillation

Use GPT-5.2 (Teacher) to generate high-quality responses, then train Qwen (Student) on them.

## Files

| File | Description |
|------|-------------|
| `generate_gpt_data.py` | Generate GPT-5.2 responses for training data |
| `train_distilled.py` | Train Step 1 model on GPT responses |

## Directories

| Directory | Description |
|-----------|-------------|
| `gpt_data/` | GPT-generated training data (created by generate_gpt_data.py) |
| `qwen3_vl_verifier_step2/` | Step 2 trained model (created by train_distilled.py) |

## Workflow

```
SFT Training Data → GPT-5.2 → Better Responses → Train Qwen → Distilled Model
```

## Usage

```bash
cd Teacher_Student_Distill

# 1. Set OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# 2. Generate GPT responses (start small)
python3 generate_gpt_data.py --num_samples 10  # Test first
python3 generate_gpt_data.py --num_samples 500  # Full run

# 3. Train on GPT data
python3 train_distilled.py \
    --base_model ../SFT/qwen3_vl_verifier/final \
    --output_dir ./qwen3_vl_verifier_distilled

# Or train 8B version
python3 train_distilled.py \
    --base_model ../SFT/qwen3_vl_verifier_8b/final \
    --output_dir ./qwen3_vl_verifier_8b_distilled \
    --lr 5e-5

# 4. Test distilled model
cd ../SFT
python3 test_verifier.py --model_path ../Teacher_Student_Distill/qwen3_vl_verifier_distilled/final
```

## GPT Model Options

| Model | Input Cost | Output Cost | Recommended |
|-------|------------|-------------|-------------|
| `gpt-5.2` (default) | $1.75/M | $14.00/M | ✅ Best for high-input tasks |
| `gpt-4o` | $2.50/M | $10.00/M | Good for balanced I/O |
| `gpt-4o-mini` | Cheaper | Cheaper | Budget option |

## Cost Estimates (GPT-5.2)

| Samples | Approx Cost |
|---------|-------------|
| 100 | ~$1-2 |
| 500 | ~$5-10 |
| 1,000 | ~$10-20 |

## Resume from Interruption

```bash
# Resume from sample 500 if interrupted
python3 generate_gpt_data.py --num_samples 500 --start_idx 500
```

