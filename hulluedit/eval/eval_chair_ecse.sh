#!/bin/bash

# CHAIR 评估脚本 - 用于评估 ECSE 生成的 caption 结果
# 使用方法: bash eval_chair_ecse.sh

# 设置环境变量
export NLTK_DATA=/data/home/scyb531/nltk_data

# 输入文件路径
INPUT_JSONL="/data/home/scyb531/lyg/HulluEdit/outputs/chair/chair_captions_ecse_val_20251107_150256.jsonl"

# COCO annotations 目录
COCO_ANNOTATIONS="/data/home/scyb531/DATA/annotations"

# CHAIR evaluator 缓存路径
CHAIR_CACHE="/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl"

# 输出文件路径（自动生成）
OUTPUT_JSON="/data/home/scyb531/lyg/HulluEdit/outputs/chair/chair_captions_ecse_val_20251107_150256_chair_result.json"

echo "========================================"
echo "CHAIR 评估 - ECSE 生成的 Captions"
echo "========================================"
echo "输入文件: $INPUT_JSONL"
echo "COCO Annotations: $COCO_ANNOTATIONS"
echo "输出文件: $OUTPUT_JSON"
echo "========================================"

# 运行评估
python /data/home/scyb531/lyg/HulluEdit/ecse/eval/eval_chair.py \
    --input "$INPUT_JSONL" \
    --coco-annotations "$COCO_ANNOTATIONS" \
    --output "$OUTPUT_JSON" \
    --cache "$CHAIR_CACHE" \
    --verbose

echo ""
echo "========================================"
echo "评估完成！"
echo "结果已保存至: $OUTPUT_JSON"
echo "========================================"

