#!/bin/bash
#SBATCH -J ecse-caption-full
#SBATCH -o logs/caption_full_%j.out
#SBATCH -e logs/caption_full_%j.err
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

# 允许通过环境变量覆盖配置与标签
CONFIG=${CONFIG:-configs/ecse_chair_eval_robust.yaml}
RUN_TAG=${RUN_TAG:-robust}

RUN_ID=$(date +%Y%m%d_%H%M%S)
CAP_JSONL=outputs/chair/chair_captions_ecse_val_${RUN_ID}_${RUN_TAG}.jsonl
CHAIR_JSON=outputs/chair/chair_captions_ecse_val_${RUN_ID}_${RUN_TAG}_chair_result.json
COCO_ANNOTATIONS=/data/home/scyb531/DATA/annotations
CHAIR_CACHE=/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl

echo "=========================================="
echo "ECSE Caption 生成（CHAIR 评测 - 500 样本）"
echo "开始时间: $(date)"
echo "输出文件: $CAP_JSONL"
echo "=========================================="

# 生成 CHAIR 所需的 JSONL captions（可参数化配置）
python -m ecse.eval.generate_chair_captions \
    --config "$CONFIG" \
    --output-file "$CAP_JSONL"

echo ""
echo "=========================================="
echo "CHAIR 指标评估"
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

# 打印摘要（含 BLEU/CHAIR）
if [ -f "$CHAIR_JSON" ]; then
    python - << 'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
metrics = data.get('overall_metrics', {})
print("==========================================")
print("结果摘要 (" + path + ")")
print("样本数:       ", data.get('num_samples', 0))
print("CHAIRs:       %.4f" % metrics.get('CHAIRs', 0.0))
print("CHAIRi:       %.4f" % metrics.get('CHAIRi', 0.0))
print("Recall:       %.4f" % metrics.get('Recall', 0.0))
print("Len:          %.4f" % metrics.get('Len', 0.0))
print("BLEU_1:       %.4f" % metrics.get('Bleu_1', 0.0))
print("BLEU_2:       %.4f" % metrics.get('Bleu_2', 0.0))
print("BLEU_3:       %.4f" % metrics.get('Bleu_3', 0.0))
print("BLEU_4:       %.4f" % metrics.get('Bleu_4', 0.0))
print("BLEU_avg:     %.4f" % metrics.get('BLEU_avg', 0.0))
print("==========================================")
PY
fi

echo ""
echo "=========================================="
echo "完成时间: $(date)"
echo "CHAIR 结果: $CHAIR_JSON"
echo "=========================================="

