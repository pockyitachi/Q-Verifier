#!/bin/bash

# ============================================
# Agent-SAMA Multi-Task Runner
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

# Task configuration
TASK_PATH="Data/rerun_issues.json"
SETTING="multi"

# Check if ADB is available
if [ -z "$ADB_PATH" ]; then
    echo "Warning: ADB_PATH not set, using 'adb' from PATH"
fi

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GEMINI_API_KEY" ] && [ -z "$CLAUDE_API_KEY" ]; then
    echo "Error: No API key found. Please set one of: OPENAI_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY"
    exit 1
fi

echo "Starting multi-task run with $TASK_PATH..."
echo "Setting: $SETTING"

python run.py \
    --run_name "multi_test" \
    --setting $SETTING \
    --tasks_json $TASK_PATH \
    --max_itr 50 \
    --collect_trajectory \
    --trajectory_output_dir "trajectory_data"

echo "Multi-task run completed!"
