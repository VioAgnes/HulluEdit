#!/bin/bash
#SBATCH -J pope-llava
#SBATCH -o /path/to/logs/pope_llava_%j.out
#SBATCH -e /path/to/logs/pope_llava_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# POPE Evaluation Script - LLaVA-1.5
# Uses ECSE method for POPE evaluation

set -euo pipefail

# Environment setup (modify according to your system)
# module load miniconda/24.9.2 || true
# module load cuda/12.1 || true
# source activate your_env || true

# Set up Python path
export PYTHONPATH=/path/to/HulluEdit:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# Configuration paths
# NOTE: Please update these paths according to your environment
CONFIG_FILE=/path/to/HulluEdit/configs/ecse_pope_llava.yaml
OUTPUT_DIR=/path/to/HulluEdit/outputs/pope_llava
HULLU_ROOT=/path/to/HulluEdit

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /path/to/HulluEdit/logs

echo "=================================="
echo "[INFO] POPE Evaluation - LLaVA-1.5 with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo ""

# Run all three splits
SPLITS=("random" "popular" "adversarial")

for SPLIT in "${SPLITS[@]}"; do
    echo "=================================="
    echo "[INFO] Processing split: $SPLIT"
    echo "=================================="

    OUTPUT_FILE="$OUTPUT_DIR/pope_llava_${SPLIT}.json"

    python $HULLU_ROOT/hulluedit/eval/pope_eval.py \
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

# Aggregate results
echo "=================================="
echo "[INFO] Aggregating results..."
echo "=================================="

python $HULLU_ROOT/hulluedit/eval/aggregate_pope.py \
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
