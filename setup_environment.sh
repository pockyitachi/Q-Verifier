#!/bin/bash
# ============================================================
# Q-Verifier Training Environment Setup
# ============================================================
# This script sets up the COMPLETE training environment.
# After running this, you can run any training script.
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

# Patch 1: unsloth_zoo/compiler.py - Add psutil import
COMPILER_FILE="$SITE_PACKAGES/unsloth_zoo/compiler.py"
if [ -f "$COMPILER_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: psutil import" "$COMPILER_FILE"; then
        echo "   ✓ compiler.py already patched"
    else
        python3 << 'EOF'
import re
import sys

compiler_file = sys.argv[1] if len(sys.argv) > 1 else "$COMPILER_FILE"
with open("$COMPILER_FILE", 'r') as f:
    content = f.read()

if 'import psutil' not in content:
    pattern = r'(check_imports\s*=\s*""")'
    replacement = r'\1\nimport psutil  # VERIFIER_PATCH: psutil import'
    
    if 'check_imports' in content:
        content = re.sub(pattern, replacement, content, count=1)
    else:
        pattern = r'(import numpy as np)'
        replacement = r'import psutil  # VERIFIER_PATCH: psutil import\n\1'
        content = re.sub(pattern, replacement, content, count=1)
    
    with open("$COMPILER_FILE", 'w') as f:
        f.write(content)
    print("   ✓ Patched compiler.py")
else:
    print("   ✓ compiler.py already has psutil")
EOF
    fi
else
    echo "   ⚠ compiler.py not found - skipping patch"
fi

# Patch 2: unsloth/models/rl.py - Fix _is_vlm detection
RL_FILE="$SITE_PACKAGES/unsloth/models/rl.py"
if [ -f "$RL_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: VLM fix" "$RL_FILE"; then
        echo "   ✓ rl.py already patched"
    else
        python3 << 'EOF'
with open("$RL_FILE", 'r') as f:
    content = f.read()

old_pattern = '''        # Temporary patch
        try:
            if model.config.model_type.endswith("_vlm"):
                self._is_vlm = False
        except:
            pass'''

new_pattern = '''        # VERIFIER_PATCH: VLM fix - commented out broken code
        # # Temporary patch
        # try:
        #     if model.config.model_type.endswith("_vlm"):
        #         self._is_vlm = False
        # except:
        #     pass'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open("$RL_FILE", 'w') as f:
        f.write(content)
    print("   ✓ Patched rl.py")
else:
    import re
    pattern = r'# Temporary patch\s+try:\s+if model\.config\.model_type\.endswith\("_vlm"\):\s+self\._is_vlm = False\s+except:\s+pass'
    if re.search(pattern, content):
        content = re.sub(pattern, '# VERIFIER_PATCH: VLM fix - removed broken temporary patch', content)
        with open("$RL_FILE", 'w') as f:
            f.write(content)
        print("   ✓ Patched rl.py (regex)")
    else:
        print("   ⚠ Pattern not found in rl.py - may already be patched")
EOF
    fi
else
    echo "   ⚠ rl.py not found - skipping patch"
fi

# Patch 3: unsloth_zoo/dataset_utils.py - Fix data collator
DATASET_FILE="$SITE_PACKAGES/unsloth_zoo/dataset_utils.py"
if [ -f "$DATASET_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: data collator fix" "$DATASET_FILE"; then
        echo "   ✓ dataset_utils.py already patched"
    else
        python3 << 'EOF'
with open("$DATASET_FILE", 'r') as f:
    content = f.read()

old_pattern = '''    if data_collator is None:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)'''

new_pattern = '''    # VERIFIER_PATCH: data collator fix - don't override VLM collator
    # if data_collator is None:
    #     from transformers import DataCollatorForLanguageModeling
    #     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open("$DATASET_FILE", 'w') as f:
        f.write(content)
    print("   ✓ Patched dataset_utils.py")
else:
    print("   ⚠ Pattern not found in dataset_utils.py - may already be patched")
EOF
    fi
else
    echo "   ⚠ dataset_utils.py not found - skipping patch"
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
echo "You can now run any training script:"
echo ""
echo "  📚 SFT Training:"
echo "     cd SFT"
echo "     python3 train_verifier.py --model unsloth/Qwen3-VL-8B-Instruct"
echo ""
echo "  🎓 Distillation (requires OPENAI_API_KEY):"
echo "     cd Teacher_Student_Distill"
echo "     export OPENAI_API_KEY='sk-...'"
echo "     python3 generate_gpt_data.py"
echo "     python3 train_distilled.py"
echo ""
echo "  🎯 DPO Training:"
echo "     cd Rejection_Sampling_DPO"
echo "     python3 train_dpo_simple.py --pairs_path dpo_pairs_488.jsonl"
echo ""
echo "  🚀 GRPO Training:"
echo "     cd GRPO"
echo "     python3 train_grpo.py"
echo ""
echo "  🧪 Testing:"
echo "     cd SFT"
echo "     python3 test_verifier.py --model_path <path_to_model>/final"
echo ""
echo "============================================================"
