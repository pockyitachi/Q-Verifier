#!/bin/bash
# Run Agent-SAMA test suite

echo "========================================"
echo "  Agent-SAMA Test Suite"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Option 1: Using unittest directly (no extra dependencies)
echo "Running tests with unittest..."
python tests/test_agents.py

# Option 2: Using pytest if available
# Uncomment below to use pytest instead:
# echo "Running tests with pytest..."
# python -m pytest tests/ -v

