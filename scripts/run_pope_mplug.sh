#!/bin/bash
#SBATCH -J pope-mplug-ecse
#SBATCH -o /path/to/logs/pope_mplug_%j.out
#SBATCH -e /path/to/logs/pope_mplug_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# POPE Evaluation Script - mPLUG-Owl2
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
CONFIG_FILE=/path/to/HulluEdit/configs/ecse_pope_mplug.yaml
OUTPUT_DIR=/path/to/HulluEdit/outputs/pope_mplug
HULLU_ROOT=/path/to/HulluEdit

# Set mPLUG-Owl2 model path (update this to your model location)
MODEL_PATH=${MODEL_PATH:-/path/to/mplug-owl2-llama2-7b}

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /path/to/HulluEdit/logs

echo "=================================="
echo "[INFO] POPE Evaluation - mPLUG-Owl2 with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Model path: $MODEL_PATH"
echo ""

# Run all three splits
SPLITS=("random" "popular" "adversarial")

for SPLIT in "${SPLITS[@]}"; do
    echo "=================================="
    echo "[INFO] Processing split: $SPLIT"
    echo "=================================="

    OUTPUT_FILE="$OUTPUT_DIR/pope_mplug_${SPLIT}.json"

    python $HULLU_ROOT/hulluedit/eval/pope_eval.py \
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

# Aggregate results
if [ -f $HULLU_ROOT/hulluedit/eval/aggregate_pope.py ]; then
    echo "=================================="
    echo "[INFO] Aggregating results..."
    echo "=================================="

    python $HULLU_ROOT/hulluedit/eval/aggregate_pope.py \
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
