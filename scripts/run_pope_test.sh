#!/bin/bash
# 快速测试 POPE 评测（10 个样本）

set -euo pipefail

# 环境设置
export PYTHONPATH=/data/home/scyb531/lyg/ParamSteer:/data/home/scyb531/lyg/HulluEdit:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# 测试 LLaVA
echo "=================================="
echo "[TEST] Testing POPE with LLaVA-1.5 (10 samples)"
echo "=================================="

python /data/home/scyb531/lyg/HulluEdit/ecse/eval/pope_eval.py \
    --config /data/home/scyb531/lyg/HulluEdit/configs/ecse_pope_llava.yaml \
    --split adversarial \
    --max-samples 10 \
    --output /data/home/scyb531/lyg/HulluEdit/outputs/pope_test_llava.json \
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

python /data/home/scyb531/lyg/HulluEdit/ecse/eval/pope_eval.py \
    --config /data/home/scyb531/lyg/HulluEdit/configs/ecse_pope_minigpt4.yaml \
    --split adversarial \
    --max-samples 10 \
    --output /data/home/scyb531/lyg/HulluEdit/outputs/pope_test_minigpt4.json \
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

