#!/bin/bash
# ============================================================
# Verifier Training Environment Setup
# ============================================================
# This script sets up the complete training environment including:
# 1. Installing dependencies
# 2. Applying necessary patches to Unsloth for VLM training
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "VERIFIER TRAINING ENVIRONMENT SETUP"
echo "============================================================"

# Step 1: Install base dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip3 install --upgrade pip

# Core training dependencies
pip3 install torch torchvision
pip3 install unsloth
pip3 install trl transformers datasets
pip3 install Pillow psutil jinja2>=3.1.0
pip3 install openai  # For GPT data generation

echo "✅ Dependencies installed"

# Step 2: Find site-packages location
echo ""
echo "[2/4] Locating site-packages..."

SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())")
if [ ! -d "$SITE_PACKAGES" ]; then
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
fi

echo "   Site-packages: $SITE_PACKAGES"

# Step 3: Apply Unsloth patches
echo ""
echo "[3/4] Applying Unsloth patches for VLM training..."

# Patch 1: unsloth_zoo/compiler.py - Add psutil import
COMPILER_FILE="$SITE_PACKAGES/unsloth_zoo/compiler.py"
if [ -f "$COMPILER_FILE" ]; then
    # Check if already patched
    if grep -q "# VERIFIER_PATCH: psutil import" "$COMPILER_FILE"; then
        echo "   ✓ compiler.py already patched"
    else
        # Find the line with compile() function and add psutil import
        python3 << EOF
import re

with open("$COMPILER_FILE", 'r') as f:
    content = f.read()

# Find where imports are added to the compiled code and add psutil
# Look for the pattern that adds imports
if 'import psutil' not in content:
    # Add psutil to the import injection
    pattern = r'(check_imports\s*=\s*""")'
    replacement = r'\1\nimport psutil  # VERIFIER_PATCH: psutil import'
    
    if 'check_imports' in content:
        content = re.sub(pattern, replacement, content, count=1)
    else:
        # Alternative: add at the top of generated code
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
    echo "   ⚠ compiler.py not found at $COMPILER_FILE"
fi

# Patch 2: unsloth/models/rl.py - Fix _is_vlm detection
RL_FILE="$SITE_PACKAGES/unsloth/models/rl.py"
if [ -f "$RL_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: VLM fix" "$RL_FILE"; then
        echo "   ✓ rl.py already patched"
    else
        python3 << EOF
with open("$RL_FILE", 'r') as f:
    content = f.read()

# Comment out the temporary patch that sets _is_vlm = False
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
    # Try alternative pattern matching
    import re
    pattern = r'# Temporary patch\s+try:\s+if model\.config\.model_type\.endswith\("_vlm"\):\s+self\._is_vlm = False\s+except:\s+pass'
    if re.search(pattern, content):
        content = re.sub(pattern, '# VERIFIER_PATCH: VLM fix - removed broken temporary patch', content)
        with open("$RL_FILE", 'w') as f:
            f.write(content)
        print("   ✓ Patched rl.py (regex)")
    else:
        print("   ⚠ Pattern not found in rl.py - may already be patched or different version")
EOF
    fi
else
    echo "   ⚠ rl.py not found at $RL_FILE"
fi

# Patch 3: unsloth_zoo/dataset_utils.py - Fix data collator
DATASET_FILE="$SITE_PACKAGES/unsloth_zoo/dataset_utils.py"
if [ -f "$DATASET_FILE" ]; then
    if grep -q "# VERIFIER_PATCH: data collator fix" "$DATASET_FILE"; then
        echo "   ✓ dataset_utils.py already patched"
    else
        python3 << EOF
with open("$DATASET_FILE", 'r') as f:
    content = f.read()

# Comment out the code that overrides VLM data collator
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
    print("   ⚠ Pattern not found in dataset_utils.py - may already be patched or different version")
EOF
    fi
else
    echo "   ⚠ dataset_utils.py not found at $DATASET_FILE"
fi

# Step 4: Clear compiled cache
echo ""
echo "[4/4] Clearing Unsloth compiled cache..."

# Find and remove cached files
CACHE_DIRS=(
    "$SITE_PACKAGES/unsloth_compiled_cache"
    "$HOME/.cache/unsloth"
    "./.unsloth_compiled_cache"
)

for cache_dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        rm -rf "$cache_dir"
        echo "   ✓ Removed $cache_dir"
    fi
done

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "You can now run training with:"
echo "  cd SFT && python3 train_verifier.py"
echo ""
echo "Or test models with:"
echo "  cd SFT && python3 test_verifier.py --model_path ./qwen3_vl_verifier/final"
echo ""

