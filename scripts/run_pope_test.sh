#!/bin/bash
# Quick POPE evaluation test (10 samples)

set -euo pipefail

# Environment setup
export PYTHONPATH=/path/to/HulluEdit:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# Test LLaVA
echo "=================================="
echo "[TEST] Testing POPE with LLaVA-1.5 (10 samples)"
echo "=================================="

python /path/to/HulluEdit/hulluedit/eval/pope_eval.py \
    --config /path/to/HulluEdit/configs/ecse_pope_llava.yaml \
    --split adversarial \
    --max-samples 10 \
    --output /path/to/HulluEdit/outputs/pope_test_llava.json \
    --model-name "LLaVA-1.5"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] LLaVA test passed"
else
    echo "[FAILED] LLaVA test failed"
    exit 1
fi

echo ""
echo "=================================="
echo "[TEST] Testing POPE with MiniGPT-4 (10 samples)"
echo "=================================="

python /path/to/HulluEdit/hulluedit/eval/pope_eval.py \
    --config /path/to/HulluEdit/configs/ecse_pope_minigpt4.yaml \
    --split adversarial \
    --max-samples 10 \
    --output /path/to/HulluEdit/outputs/pope_test_minigpt4.json \
    --model-name "MiniGPT-4"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] MiniGPT-4 test passed"
else
    echo "[FAILED] MiniGPT-4 test failed"
    exit 1
fi

echo ""
echo "=================================="
echo "[SUCCESS] All tests passed!"
echo "=================================="
