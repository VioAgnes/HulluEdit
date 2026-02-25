#!/bin/bash
#SBATCH -J pope-mplug-ecse
#SBATCH -o /data/home/scyb531/lyg/HulluEdit/logs/pope_mplug_%j.out
#SBATCH -e /data/home/scyb531/lyg/HulluEdit/logs/pope_mplug_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# POPE 评测脚本 - mPLUG-Owl2
# 使用 ECSE 方法进行 POPE 评测

set -euo pipefail

# 环境加载
module load miniconda/24.9.2 || true
module load cuda/12.1 || true
source activate deco311 || true

# 环境设置
# 使用 ParamSteer 的 transformers 库来解决版本兼容问题
export PYTHONPATH=/data/home/scyb531/lyg/ParamSteer:/data/home/scyb531/lyg/HulluEdit:/data/home/scyb531/sjc/ECSE:/data/home/scyb531/sjc/Nullu:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# 配置路径
CONFIG_FILE=/data/home/scyb531/lyg/HulluEdit/configs/ecse_pope_mplug.yaml
OUTPUT_DIR=/data/home/scyb531/lyg/HulluEdit/outputs/pope_mplug

# 请设置 mPLUG-Owl2 HF 或本地权重路径（如果配置文件中未设置）
MODEL_PATH=${MODEL_PATH:-/data/home/scyb531/MODELS/mplug-owl2-llama2-7b}

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p /data/home/scyb531/lyg/HulluEdit/logs

echo "=================================="
echo "[INFO] POPE Evaluation - mPLUG-Owl2 with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Model path: $MODEL_PATH"
echo ""

# 运行所有三个 split
SPLITS=("random" "popular" "adversarial")

for SPLIT in "${SPLITS[@]}"; do
    echo "=================================="
    echo "[INFO] Processing split: $SPLIT"
    echo "=================================="
    
    OUTPUT_FILE="$OUTPUT_DIR/pope_mplug_${SPLIT}.json"
    
    python /data/home/scyb531/lyg/HulluEdit/ecse/eval/pope_eval.py \
        --config "$CONFIG_FILE" \
        --split "$SPLIT" \
        --output "$OUTPUT_FILE" \
        --model-name "mPLUG-Owl2" \
        --model-path "$MODEL_PATH"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] POPE evaluation failed for split: $SPLIT"
        exit 1
    fi
    
    echo "[INFO] Completed split: $SPLIT"
    echo ""
done

# 聚合结果（如果存在聚合脚本）
if [ -f /data/home/scyb531/lyg/HulluEdit/ecse/eval/aggregate_pope.py ]; then
    echo "=================================="
    echo "[INFO] Aggregating results..."
    echo "=================================="
    
    python /data/home/scyb531/lyg/HulluEdit/ecse/eval/aggregate_pope.py \
        --files \
            "$OUTPUT_DIR/pope_mplug_random.json" \
            "$OUTPUT_DIR/pope_mplug_popular.json" \
            "$OUTPUT_DIR/pope_mplug_adversarial.json" \
        --output "$OUTPUT_DIR/pope_mplug_all_metrics.json"
    
    if [ $? -ne 0 ]; then
        echo "[WARNING] Aggregation failed, but individual results are available"
    fi
fi

echo ""
echo "=================================="
echo "[INFO] Evaluation completed successfully!"
echo "=================================="
echo "[INFO] Results location:"
echo "  - Random: $OUTPUT_DIR/pope_mplug_random.json"
echo "  - Popular: $OUTPUT_DIR/pope_mplug_popular.json"
echo "  - Adversarial: $OUTPUT_DIR/pope_mplug_adversarial.json"
if [ -f "$OUTPUT_DIR/pope_mplug_all_metrics.json" ]; then
    echo "  - Aggregated: $OUTPUT_DIR/pope_mplug_all_metrics.json"
fi
echo ""
echo "[INFO] End time: $(date)"
echo "=================================="

