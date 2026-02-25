#!/bin/bash

# 使用现有 eval_chair.py 评估 ECSE 生成的 captions
# 使用方法: bash eval_ecse_captions.sh

# 设置环境变量
export NLTK_DATA=/data/home/scyb531/nltk_data

# 切换到脚本目录
cd "$(dirname "$0")"

# 输入文件
INPUT_JSONL="/data/home/scyb531/lyg/HulluEdit/outputs/chair/chair_captions_ecse_val_20251107_150256.jsonl"

# COCO annotations 目录
COCO_ANNOTATIONS="/data/home/scyb531/DATA/annotations"

# CHAIR evaluator 缓存
CHAIR_CACHE="/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl"

# 输出文件
OUTPUT_JSON="/data/home/scyb531/lyg/HulluEdit/outputs/chair/ecse_chair_eval_results.json"

echo "========================================"
echo "CHAIR 评估 - ECSE Captions"
echo "========================================"
echo "输入: $INPUT_JSONL"
echo "输出: $OUTPUT_JSON"
echo "========================================"
echo ""

# 运行评估
python eval_chair.py \
    --input "$INPUT_JSONL" \
    --coco-annotations "$COCO_ANNOTATIONS" \
    --output "$OUTPUT_JSON" \
    --cache "$CHAIR_CACHE" \
    --verbose

echo ""
echo "========================================"
echo "评估完成！"
echo "========================================"

