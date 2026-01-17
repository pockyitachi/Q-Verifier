#!/bin/bash

# ============================================
# Agent-SAMA Single-Task Runner
# ============================================
# 
# Prerequisites (export these before running):
#   export ADB_PATH="/path/to/adb"
#   export BACKBONE_TYPE="OpenAI"  # or "Gemini" or "Claude"
#   export OPENAI_API_KEY="sk-..."  # or GEMINI_API_KEY / CLAUDE_API_KEY
#   export QWEN_API_KEY="sk-..."    # for image captioning
#
# Make sure your phone is connected via ADB:
#   adb devices
# ============================================

set -e  # Exit on error

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GEMINI_API_KEY" ] && [ -z "$CLAUDE_API_KEY" ]; then
    echo "Error: No API key found. Please set one of: OPENAI_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY"
    exit 1
fi

echo "Starting single-task run..."

python run.py \
    --run_name "single_test" \
    --setting "single" \
    --instruction "Find the best-rated late-night Korean restaurant in Champaign, IL that opens beyond 9pm on Google Maps." \
    --max_itr 30 \
    --collect_trajectory \
    --trajectory_output_dir "trajectory_data"

echo "Single-task run completed!"
