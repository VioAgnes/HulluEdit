#!/bin/bash
#SBATCH -J ecse-pope-full
#SBATCH -o logs/pope_full_%j.out
#SBATCH -e logs/pope_full_%j.err
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

# 创建日志目录
mkdir -p logs

# 运行 POPE 评测（完整数据集）
echo "=========================================="
echo "ECSE POPE 完整评测"
echo "开始时间: $(date)"
echo "=========================================="

# Adversarial split (最难)
python -m ecse.eval.pope_eval \
    --config configs/ecse_llava7b.yaml \
    --split adversarial \
    --output outputs/pope_adversarial.json

echo ""
echo "=========================================="
echo "评测完成: $(date)"
echo "=========================================="

