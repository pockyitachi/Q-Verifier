# SFT: Supervised Fine-Tuning

Train Qwen3-VL verifier on trajectory data clustered by subgoals.

## Files

| File | Description |
|------|-------------|
| `train_verifier.py` | Main training script |
| `prepare_training_data.py` | Prepare raw trajectory data |
| `prepare_training_data_filtered.py` | Prepare filtered trajectory data |
| `test_verifier.py` | Test fine-tuned model |
| `test_base_model.py` | Test base model (no fine-tuning) |

## Directories

| Directory | Description |
|-----------|-------------|
| `training_data_filtered/` | Training data with images |
| `qwen3_vl_verifier/` | Trained 4B model |
| `qwen3_vl_verifier_8b/` | Trained 8B model |

## Usage

```bash
cd SFT

# Train 4B model
python3 train_verifier.py \
    --data_dir training_data_filtered \
    --model unsloth/Qwen3-VL-4B-Instruct \
    --epochs 3 \
    --output_dir ./qwen3_vl_verifier

# Train 8B model
python3 train_verifier.py \
    --data_dir training_data_filtered \
    --model unsloth/Qwen3-VL-8B-Instruct \
    --epochs 3 \
    --output_dir ./qwen3_vl_verifier_8b

# Test fine-tuned model
python3 test_verifier.py --model_path ./qwen3_vl_verifier/final --num_samples 50

# Test base model (comparison)
python3 test_base_model.py --num_samples 50
```

## Trained Models

- **4B Model**: `qwen3_vl_verifier/final/`
- **8B Model**: `qwen3_vl_verifier_8b/final/`

## Metrics

- **SR(Traj)**: Success rate at trajectory level (correct predictions)
- **Tokens**: Input/output token counts per sample

