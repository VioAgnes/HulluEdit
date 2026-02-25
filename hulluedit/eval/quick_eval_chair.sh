#!/bin/bash

# 快速运行 CHAIR 评估
# 使用方法: bash quick_eval_chair.sh

export NLTK_DATA=/data/home/scyb531/nltk_data

cd /data/home/scyb531/lyg/HulluEdit/ecse/eval

python run_chair_eval.py \
    --input /data/home/scyb531/lyg/HulluEdit/outputs/chair/chair_captions_ecse_val_20251107_150256.jsonl \
    --verbose

