#!/bin/bash
# ============================================================
# Q-Verifier Training Environment Setup
# ============================================================
# Original paper training environment recipe.
# Later BF16/v9 workflows use their own recorded environments.
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# Requirements:
#   - Python 3.10+
#   - NVIDIA GPU with CUDA 11.8+ (24GB+ VRAM recommended)
#   - ~50GB disk space for models and cache
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Q-VERIFIER TRAINING ENVIRONMENT SETUP"
echo "============================================================"

# Step 0: Check prerequisites
echo ""
echo "[0/5] Checking prerequisites..."

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "   ❌ Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "   ✓ Python $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo "   ✓ GPU: $GPU_INFO"
else
    echo "   ⚠ nvidia-smi not found - GPU training may not work"
fi

# Step 1: Install base dependencies
echo ""
echo "[1/5] Installing Python dependencies..."
pip3 install --upgrade pip

# Get script directory (where requirements.txt is)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install from requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "   Installing from requirements.txt..."
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "   ⚠ requirements.txt not found, installing manually..."
    # Fallback: install packages individually
    pip3 install torch torchvision
    pip3 install unsloth
    pip3 install transformers>=4.45.0 datasets>=2.14.0 accelerate>=0.25.0
    pip3 install peft>=0.7.0 trl>=0.12.0 bitsandbytes>=0.41.0
    pip3 install safetensors huggingface_hub
    pip3 install Pillow>=10.0.0 qwen-vl-utils
    pip3 install psutil jinja2>=3.1.0 sentencepiece protobuf openai
fi

echo "✅ Dependencies installed"

# Step 2: Find site-packages location
echo ""
echo "[2/5] Locating site-packages..."

SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ ! -d "$SITE_PACKAGES" ]; then
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
fi

echo "   Site-packages: $SITE_PACKAGES"

# Step 3: Apply Unsloth patches for VLM support
echo ""
echo "[3/5] Applying Unsloth patches for VLM training..."

# Main patch: unsloth/models/rl.py - Fix _is_vlm detection bug
# Unsloth 2026.x has code that replaces "_is_vlm = True" with "_is_vlm = False"
# This breaks VLM training. We comment out that buggy code.
RL_FILE="$SITE_PACKAGES/unsloth/models/rl.py"
if [ -f "$RL_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: VLM fix" "$RL_FILE"; then
        echo "   ✓ rl.py already patched"
    else
        # Use Python with the file path passed as argument
        python3 - "$RL_FILE" << 'PATCH_EOF'
import sys

rl_file = sys.argv[1]
print(f"   Patching {rl_file}...")

with open(rl_file, 'r') as f:
    content = f.read()

# Pattern for Unsloth 2026.x - the buggy code that replaces _is_vlm = True with False
old_pattern = '''        # Temporary patch _is_vlm to False
        # as of 0.22 it only exists in sfttrainer
        original_is_vlm_text = "self._is_vlm = True"
        new_is_vlm_text = "self._is_vlm = False"
        RLTrainer_source = RLTrainer_source.replace(
            original_is_vlm_text, new_is_vlm_text
        )'''

new_pattern = '''        # VERIFIER_PATCH: VLM fix - commented out buggy code
        # The original code incorrectly sets _is_vlm = False for VLM models
        # # Temporary patch _is_vlm to False
        # # as of 0.22 it only exists in sfttrainer
        # original_is_vlm_text = "self._is_vlm = True"
        # new_is_vlm_text = "self._is_vlm = False"
        # RLTrainer_source = RLTrainer_source.replace(
        #     original_is_vlm_text, new_is_vlm_text
        # )
        pass  # Keep _is_vlm = True for VLM support'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open(rl_file, 'w') as f:
        f.write(content)
    print("   ✓ Patched rl.py - fixed VLM detection bug")
elif "# VERIFIER_PATCH: VLM fix" in content:
    print("   ✓ rl.py already patched")
else:
    # Try alternative pattern (older Unsloth versions)
    alt_pattern = 'original_is_vlm_text = "self._is_vlm = True"'
    if alt_pattern in content:
        # Comment out just the replacement lines
        content = content.replace(
            'original_is_vlm_text = "self._is_vlm = True"',
            '# VERIFIER_PATCH: VLM fix\n        # original_is_vlm_text = "self._is_vlm = True"'
        )
        content = content.replace(
            'new_is_vlm_text = "self._is_vlm = False"',
            '# new_is_vlm_text = "self._is_vlm = False"'
        )
        content = content.replace(
            'RLTrainer_source = RLTrainer_source.replace(\n            original_is_vlm_text, new_is_vlm_text\n        )',
            '# RLTrainer_source = RLTrainer_source.replace(...)\n        pass  # Keep _is_vlm = True'
        )
        with open(rl_file, 'w') as f:
            f.write(content)
        print("   ✓ Patched rl.py (alternative pattern)")
    else:
        print("   ⚠ VLM bug pattern not found - may be a different Unsloth version")
PATCH_EOF
    fi
else
    echo "   ⚠ rl.py not found at $RL_FILE - skipping patch"
fi

# Step 4: Clear compiled cache
echo ""
echo "[4/5] Clearing Unsloth compiled cache..."

CACHE_DIRS=(
    "$SITE_PACKAGES/unsloth_compiled_cache"
    "$HOME/.cache/unsloth"
    "./.unsloth_compiled_cache"
    "./unsloth_compiled_cache"
)

for cache_dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        rm -rf "$cache_dir"
        echo "   ✓ Removed $cache_dir"
    fi
done

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying installation..."

python3 << 'EOF'
import sys
errors = []

# Check critical imports
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠ CUDA not available - will use CPU (slow)")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import transformers
    print(f"   ✓ Transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"transformers: {e}")

try:
    import trl
    print(f"   ✓ TRL {trl.__version__}")
except ImportError as e:
    errors.append(f"trl: {e}")

try:
    import unsloth
    print(f"   ✓ Unsloth installed")
except ImportError as e:
    errors.append(f"unsloth: {e}")

try:
    from PIL import Image
    print(f"   ✓ Pillow installed")
except ImportError as e:
    errors.append(f"Pillow: {e}")

if errors:
    print("\n❌ Some packages failed to import:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
EOF

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Original paper entry points (from the repository root):"
echo ""
echo "  📚 SFT Training:"
echo "     cd training/SFT"
echo "     python3 train_verifier.py --model unsloth/Qwen3-VL-8B-Instruct"
echo ""
echo "  🎓 Distillation (requires OPENAI_API_KEY):"
echo "     cd training/Teacher_Student_Distill"
echo "     export OPENAI_API_KEY='sk-...'"
echo "     python3 generate_gpt_data.py"
echo "     python3 train_distilled.py"
echo ""
echo "  🎯 DPO Training:"
echo "     cd training/Rejection_Sampling_DPO"
echo "     Read README.md: the original trainer and current frozen pairs have different loss/data contracts."
echo ""
echo "  🚀 GRPO Training:"
echo "     cd training/GRPO"
echo "     python3 train_grpo.py"
echo ""
echo "  🧪 Testing:"
echo "     cd training/SFT"
echo "     python3 test_verifier.py --model_path <path_to_model>/final"
echo ""
echo "============================================================"
