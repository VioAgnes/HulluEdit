#!/bin/bash
#SBATCH -J ecse-pope-test
#SBATCH -o logs/pope_test_%j.out
#SBATCH -e logs/pope_test_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# Environment setup (modify according to your system)
# module load miniconda/24.9.2
# module load cuda/12.1
# source activate your_env

export PYTHONUNBUFFERED=1
export PYTHONPATH=/path/to/HulluEdit:$PYTHONPATH

# Working directory
cd /path/to/HulluEdit

# Create log directory
mkdir -p logs

# Run POPE evaluation (10 samples quick test)
echo "=========================================="
echo "ECSE POPE Quick Test (10 samples)"
echo "Start time: $(date)"
echo "=========================================="

python -m hulluedit.eval.pope_eval \
    --config configs/ecse_pope_llava.yaml \
    --split adversarial \
    --max-samples 10 \
    --output outputs/pope_test_10samples.json

echo ""
echo "=========================================="
echo "Test completed: $(date)"
echo "=========================================="
