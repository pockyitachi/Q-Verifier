# SFT: Supervised Fine-Tuning

Train Qwen VL verifiers on trajectory data using LoRA fine-tuning.

## Supported Models

| Model | Parameter | Command |
|-------|-----------|---------|
| Qwen3-VL-4B | `unsloth/Qwen3-VL-4B-Instruct` | Default |
| Qwen3-VL-8B | `unsloth/Qwen3-VL-8B-Instruct` | `--model unsloth/Qwen3-VL-8B-Instruct` |
| Qwen2.5-VL-3B | `unsloth/Qwen2.5-VL-3B-Instruct` | `--model unsloth/Qwen2.5-VL-3B-Instruct` |
| Qwen2.5-VL-7B | `unsloth/Qwen2.5-VL-7B-Instruct` | `--model unsloth/Qwen2.5-VL-7B-Instruct` |

## Files

| File | Description |
|------|-------------|
| `train_verifier.py` | Main training script (supports all models) |
| `test_verifier.py` | Evaluation script |
| `test_base_model.py` | Test base model (no fine-tuning) |
| `prepare_training_data.py` | Prepare raw trajectory data |
| `prepare_training_data_filtered.py` | Prepare filtered trajectory data |

## Trained Models

| Model | Directory |
|-------|-----------|
| Qwen3-VL-4B | `qwen3_vl_verifier_4b/final/` |
| Qwen3-VL-8B | `qwen3_vl_verifier_8b/final/` |
| Qwen2.5-VL-3B | `qwen25_vl_verifier_3b/final/` |
| Qwen2.5-VL-7B | `qwen25_vl_verifier_7b/final/` |

## Usage

### Training

```bash
cd SFT

# Train Qwen3-VL-4B (default)
python3 train_verifier.py --output_dir ./qwen3_vl_verifier_4b

# Train Qwen3-VL-8B
python3 train_verifier.py \
    --model unsloth/Qwen3-VL-8B-Instruct \
    --output_dir ./qwen3_vl_verifier_8b

# Train Qwen2.5-VL-3B
python3 train_verifier.py \
    --model unsloth/Qwen2.5-VL-3B-Instruct \
    --output_dir ./qwen25_vl_verifier_3b

# Train Qwen2.5-VL-7B
python3 train_verifier.py \
    --model unsloth/Qwen2.5-VL-7B-Instruct \
    --output_dir ./qwen25_vl_verifier_7b
```

### Testing

```bash
# Test any trained model
python3 test_verifier.py --model_path ./qwen25_vl_verifier_3b/final --num_samples 71

# Full validation set
python3 test_verifier.py --model_path ./qwen3_vl_verifier_4b/final --num_samples 71

# Save results to specific file
python3 test_verifier.py --model_path ./qwen25_vl_verifier_7b/final --output_file test_results_qwen25_7b.json
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 1 | Per-device batch size |
| `--grad_accum` | 8 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank |
| `--max_seq_length` | 4096 | Maximum sequence length |

## Data

Training data is stored in `training_data_filtered/`:
- `train.jsonl` - Training samples
- `val.jsonl` - Validation samples
- `images/` - BEFORE and AFTER screenshots

## Metrics

| Metric | Description |
|--------|-------------|
| **SR(Traj)** | Success rate on trajectory validation |
| **Tokens (Input)** | Average input tokens per sample |
| **Tokens (Output)** | Average output tokens per sample |

## Test Results

Results are saved to JSON files with format `test_results_<model>.json`:
- `test_results_4b.json` - Qwen3-VL-4B SFT
- `test_results_8b.json` - Qwen3-VL-8B SFT
- `test_results_qwen25_3b.json` - Qwen2.5-VL-3B SFT
- `test_results_qwen25_7b.json` - Qwen2.5-VL-7B SFT
