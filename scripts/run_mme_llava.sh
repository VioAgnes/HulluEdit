#!/bin/bash
#SBATCH -J mme-llava
#SBATCH -o /data/home/scyb531/lyg/HulluEdit/logs/mme_llava_%j.out
#SBATCH -e /data/home/scyb531/lyg/HulluEdit/logs/mme_llava_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# MME 评测脚本 - LLaVA-7B
# 参考 DeCo/run_mme.sh 的实现
# 使用 ECSE 方法进行 MME 评测

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
CONFIG_FILE=/data/home/scyb531/lyg/HulluEdit/configs/ecse_mme_llava.yaml
QUESTION_FILE=/data/home/scyb531/DATA/llava_hallu_mme.jsonl
IMAGE_FOLDER=/data/home/scyb531/DATA/MME_Benchmark_release_version
OUTPUT_DIR=/data/home/scyb531/lyg/HulluEdit/outputs/mme_llava

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p /data/home/scyb531/lyg/HulluEdit/logs

echo "=================================="
echo "[INFO] MME Evaluation - LLaVA-7B with ECSE"
echo "[INFO] Start time: $(date)"
echo "=================================="
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Question file: $QUESTION_FILE"
echo "[INFO] Image folder: $IMAGE_FOLDER"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo ""

# 步骤1: 生成答案
echo "[STEP 1/2] Generating answers with ECSE..."
python /data/home/scyb531/lyg/HulluEdit/ecse/eval/mme_eval.py \
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

# 步骤2: 转换答案格式
echo "[STEP 2/2] Converting answers to MME format..."
ANSWER_FILE=$(ls -t "$OUTPUT_DIR"/llava*_mme_answers_*.jsonl | head -n 1)
if [ -z "$ANSWER_FILE" ]; then
    echo "[ERROR] 未找到生成的答案文件"
    exit 1
fi

echo "[INFO] Processing file: $(basename $ANSWER_FILE)"
python /data/home/scyb531/lyg/HulluEdit/ecse/eval/convert_answer_to_mme.py \
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

# 显示结果
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

