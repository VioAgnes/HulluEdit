#!/bin/bash
#SBATCH -J mme-llava
#SBATCH -o /path/to/logs/mme_llava_%j.out
#SBATCH -e /path/to/logs/mme_llava_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# MME Evaluation Script - LLaVA-7B
# Uses ECSE method for MME evaluation

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
CONFIG_FILE=/path/to/HulluEdit/configs/ecse_mme_llava.yaml
QUESTION_FILE=/path/to/data/llava_hallu_mme.jsonl
IMAGE_FOLDER=/path/to/data/MME_Benchmark_release_version
OUTPUT_DIR=/path/to/HulluEdit/outputs/mme_llava
HULLU_ROOT=/path/to/HulluEdit

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /path/to/HulluEdit/logs

echo "=================================="
echo "[INFO] MME Evaluation - LLaVA-7B with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Question file: $QUESTION_FILE"
echo "[INFO] Image folder: $IMAGE_FOLDER"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Generate answers
echo "[STEP 1/2] Generating answers with ECSE..."
python $HULLU_ROOT/hulluedit/eval/mme_eval.py \
  --model_name LLaVA-7B \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --config "$CONFIG_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

if [ $? -ne 0 ]; then
    echo "[ERROR] Answer generation failed"
    exit 1
fi

echo ""
echo "[STEP 1/2] Answer generation completed successfully"
echo ""

# Step 2: Convert answers to MME format
echo "[STEP 2/2] Converting answers to MME format..."
ANSWER_FILE=$(ls -t "$OUTPUT_DIR"/llava*_mme_answers_*.jsonl | head -n 1)
if [ -z "$ANSWER_FILE" ]; then
    echo "[ERROR] Generated answer file not found"
    exit 1
fi

echo "[INFO] Processing file: $(basename $ANSWER_FILE)"
python $HULLU_ROOT/hulluedit/eval/convert_answer_to_mme.py \
  --output_path "$ANSWER_FILE" \
  --log_path "$OUTPUT_DIR/results" \
  --seed 42

if [ $? -ne 0 ]; then
    echo "[ERROR] Answer conversion failed"
    exit 1
fi

echo ""
echo "[STEP 2/2] Answer conversion completed successfully"
echo ""

# Display results
echo "=================================="
echo "[INFO] Evaluation completed successfully!"
echo "=================================="
echo "[INFO] Results location:"
echo "  - Answers: $ANSWER_FILE"
echo "  - Converted: $OUTPUT_DIR/results/"
echo ""
echo "[INFO] To view results:"
echo "  ls -lh $OUTPUT_DIR/results/"
echo ""
echo "[INFO] End time: $(date)"
echo "=================================="
