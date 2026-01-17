#!/bin/bash

# ============================================
# Run Agent-SAMA on Android World with Trajectory Collection
# ============================================
#
# Prerequisites:
#   1. Android emulator running (see instructions below)
#   2. API key set: export OPENAI_API_KEY="sk-..."
#
# Start emulator:
#   ~/Library/Android/sdk/emulator/emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554
#
# ============================================

set -e

# ===== Configuration =====
AGENT_NAME="${AGENT_NAME:-agent_sama}"
SUITE_FAMILY="${SUITE_FAMILY:-android_world}"
N_TASK_COMBINATIONS="${N_TASK_COMBINATIONS:-1}"
COLLECT_TRAJECTORY="${COLLECT_TRAJECTORY:-true}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-$HOME/android_world/trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/android_world/runs}"

# ===== Validation =====
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "   Run: export OPENAI_API_KEY=\"sk-...\""
    exit 1
fi

# Check if adb is available
if ! command -v adb &> /dev/null; then
    ADB_PATH="$HOME/Library/Android/sdk/platform-tools/adb"
    if [ ! -f "$ADB_PATH" ]; then
        echo "❌ Error: adb not found"
        echo "   Install Android SDK or set ADB_PATH"
        exit 1
    fi
else
    ADB_PATH=$(which adb)
fi

# Check if device is connected
if ! $ADB_PATH devices | grep -q "device$"; then
    echo "❌ Error: No Android device/emulator connected"
    echo ""
    echo "   Start an emulator with:"
    echo "   ~/Library/Android/sdk/emulator/emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554"
    echo ""
    echo "   Or connect a physical device via USB with ADB debugging enabled"
    exit 1
fi

echo "✅ Device connected:"
$ADB_PATH devices

# ===== Install dependencies =====
echo ""
echo "📦 Installing android_world..."
cd "$(dirname "$0")"
pip install -e . -q 2>/dev/null || pip install -e .

# ===== Run Agent-SAMA =====
echo ""
echo "🚀 Running Agent-SAMA on Android World"
echo "   Agent: $AGENT_NAME"
echo "   Suite: $SUITE_FAMILY"
echo "   Tasks per template: $N_TASK_COMBINATIONS"
echo "   Collect trajectory: $COLLECT_TRAJECTORY"
echo "   Trajectory output: $TRAJECTORY_DIR"
echo ""

COLLECT_FLAG=""
if [ "$COLLECT_TRAJECTORY" = "true" ]; then
    COLLECT_FLAG="--collect_trajectory --trajectory_output_dir $TRAJECTORY_DIR"
fi

python run.py \
    --agent_name "$AGENT_NAME" \
    --suite_family "$SUITE_FAMILY" \
    --n_task_combinations "$N_TASK_COMBINATIONS" \
    --output_path "$OUTPUT_DIR" \
    --adb_path "$ADB_PATH" \
    $COLLECT_FLAG

echo ""
echo "✅ Done!"
echo "   Results saved to: $OUTPUT_DIR"
if [ "$COLLECT_TRAJECTORY" = "true" ]; then
    echo "   Trajectories saved to: $TRAJECTORY_DIR"
fi
