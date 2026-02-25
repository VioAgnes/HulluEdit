#!/bin/bash
#SBATCH -J ecse-pope-all
#SBATCH -o logs/pope_all_%j.out
#SBATCH -e logs/pope_all_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

# 环境加载
module load miniconda/24.9.2
module load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate deco311
export PYTHONUNBUFFERED=1

# 使用 ParamSteer 的 transformers 库
export PYTHONPATH=/data/home/scyb531/lyg/ParamSteer:$PYTHONPATH

# 工作目录
cd /data/home/scyb531/lyg/ECSE

# 目录
mkdir -p logs outputs

echo "=========================================="
echo "ECSE POPE 三个split完整评测"
echo "开始时间: $(date)"
echo "=========================================="

for SPLIT in random popular adversarial; do
  echo "[POPE] Split: $SPLIT"
  python -m ecse.eval.pope_eval \
    --config configs/ecse_llava7b.yaml \
    --split $SPLIT \
    --output outputs/pope_${SPLIT}.json
done

echo "\n[POPE] 汇总指标"
python -m ecse.eval.aggregate_pope \
  --files outputs/pope_random.json outputs/pope_popular.json outputs/pope_adversarial.json \
  --output outputs/pope_all_metrics.json

echo ""
echo "=========================================="
echo "评测完成: $(date)"
echo "=========================================="


