#!/bin/bash
#SBATCH -J pope-llava
#SBATCH -o /data/home/scyb531/lyg/HulluEdit/logs/pope_llava_%j.out
#SBATCH -e /data/home/scyb531/lyg/HulluEdit/logs/pope_llava_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# POPE 评测脚本 - LLaVA-1.5
# 使用 ECSE 方法进行 POPE 评测

set -euo pipefail

# 环境加载
module load miniconda/24.9.2 || true
module load cuda/12.1 || true
source activate deco311 || true

# 环境设置
# 使用 ParamSteer 的 transformers 库来解决版本兼容问题
export PYTHONPATH=/data/home/scyb531/lyg/ParamSteer:/data/home/scyb531/lyg/HulluEdit:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# 配置路径
CONFIG_FILE=/data/home/scyb531/lyg/HulluEdit/configs/ecse_pope_llava.yaml
OUTPUT_DIR=/data/home/scyb531/lyg/HulluEdit/outputs/pope_llava

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p /data/home/scyb531/lyg/HulluEdit/logs

echo "=================================="
echo "[INFO] POPE Evaluation - LLaVA-1.5 with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo ""

# 运行所有三个 split
SPLITS=("random" "popular" "adversarial")

for SPLIT in "${SPLITS[@]}"; do
    echo "=================================="
    echo "[INFO] Processing split: $SPLIT"
    echo "=================================="
    
    OUTPUT_FILE="$OUTPUT_DIR/pope_llava_${SPLIT}.json"
    
    python /data/home/scyb531/lyg/HulluEdit/ecse/eval/pope_eval.py \
        --config "$CONFIG_FILE" \
        --split "$SPLIT" \
        --output "$OUTPUT_FILE" \
        --model-name "LLaVA-1.5"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] POPE evaluation failed for split: $SPLIT"
        exit 1
    fi
    
    echo "[INFO] Completed split: $SPLIT"
    echo ""
done

# 聚合结果
echo "=================================="
echo "[INFO] Aggregating results..."
echo "=================================="

python /data/home/scyb531/lyg/HulluEdit/ecse/eval/aggregate_pope.py \
    --files \
        "$OUTPUT_DIR/pope_llava_random.json" \
        "$OUTPUT_DIR/pope_llava_popular.json" \
        "$OUTPUT_DIR/pope_llava_adversarial.json" \
    --output "$OUTPUT_DIR/pope_llava_all_metrics.json"

if [ $? -ne 0 ]; then
    echo "[ERROR] Aggregation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "[INFO] Evaluation completed successfully!"
echo "=================================="
echo "[INFO] Results location:"
echo "  - Random: $OUTPUT_DIR/pope_llava_random.json"
echo "  - Popular: $OUTPUT_DIR/pope_llava_popular.json"
echo "  - Adversarial: $OUTPUT_DIR/pope_llava_adversarial.json"
echo "  - Aggregated: $OUTPUT_DIR/pope_llava_all_metrics.json"
echo ""
echo "[INFO] End time: $(date)"
echo "=================================="

