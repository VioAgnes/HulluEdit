#!/bin/bash
#SBATCH -J ecse-pope-test
#SBATCH -o logs/pope_test_%j.out
#SBATCH -e logs/pope_test_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# 环境加载
module load miniconda/24.9.2
module load cuda/12.1
source activate deco311
export PYTHONUNBUFFERED=1

# 使用 ParamSteer 的 transformers 库
export PYTHONPATH=/data/home/scyb531/lyg/ParamSteer:$PYTHONPATH

# 工作目录
cd /data/home/scyb531/lyg/ECSE

# 创建日志目录
mkdir -p logs

# 运行 POPE 评测（10 样本快速测试）
echo "=========================================="
echo "ECSE POPE 快速测试 (10 样本)"
echo "开始时间: $(date)"
echo "=========================================="

python -m ecse.eval.pope_eval \
    --config configs/ecse_llava7b.yaml \
    --split adversarial \
    --max-samples 10 \
    --output outputs/pope_test_10samples.json

echo ""
echo "=========================================="
echo "测试完成: $(date)"
echo "=========================================="

