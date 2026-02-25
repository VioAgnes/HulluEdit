#!/bin/bash
#SBATCH -J ecse-caption-test
#SBATCH -o logs/caption_test_%j.out
#SBATCH -e logs/caption_test_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# 环境加载
module load miniconda/24.9.2
module load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate deco311
export PYTHONUNBUFFERED=1

# NLTK 数据目录（用于 CHAIR 评估）
export NLTK_DATA=/data/home/scyb531/nltk_data

# Python 模块路径（ParamSteer + 当前项目）
export PYTHONPATH=/data/home/scyb531/lyg/HulluEdit:/data/home/scyb531/lyg/ParamSteer:$PYTHONPATH

# 工作目录：HulluEdit 项目根目录
cd /data/home/scyb531/lyg/HulluEdit

# 目录与路径
mkdir -p logs outputs/chair

RUN_ID=$(date +%Y%m%d_%H%M%S)
CAP_JSONL=outputs/chair/chair_captions_ecse_test_${RUN_ID}.jsonl
CHAIR_JSON=outputs/chair/chair_captions_ecse_test_${RUN_ID}_chair_result.json
COCO_ANNOTATIONS=/data/home/scyb531/DATA/annotations
CHAIR_CACHE=/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl

echo "=========================================="
echo "ECSE Caption 快速测试 (10 样本)"
echo "开始时间: $(date)"
echo "输出文件: $CAP_JSONL"
echo "=========================================="

# 生成 10 条 JSONL captions 供 CHAIR 评估使用
python -m ecse.eval.generate_chair_captions \
    --config configs/ecse_chair_eval.yaml \
    --num-samples 10 \
    --output-file "$CAP_JSONL"

echo ""
echo "=========================================="
echo "CHAIR 指标评估（测试）"
echo "输入: $CAP_JSONL"
echo "输出: $CHAIR_JSON"
echo "=========================================="

# 按 eval_chair_ecse.sh 的实现方式运行 CHAIR 评估
python -m ecse.eval.eval_chair \
    --input "$CAP_JSONL" \
    --coco-annotations "$COCO_ANNOTATIONS" \
    --output "$CHAIR_JSON" \
    --cache "$CHAIR_CACHE" \
    --verbose

echo ""
echo "=========================================="
echo "测试完成: $(date)"
echo "CHAIR 结果: $CHAIR_JSON"
echo "=========================================="

