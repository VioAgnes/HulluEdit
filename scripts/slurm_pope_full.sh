#!/bin/bash
#SBATCH -J ecse-pope-full
#SBATCH -o logs/pope_full_%j.out
#SBATCH -e logs/pope_full_%j.err
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

# Environment setup (modify according to your system)
# module load miniconda/24.9.2
# module load cuda/12.1
# eval "$(conda shell.bash hook)"
# conda activate your_env

export PYTHONUNBUFFERED=1
export PYTHONPATH=/path/to/HulluEdit:$PYTHONPATH

# Working directory
cd /path/to/HulluEdit

# Create log directory
mkdir -p logs

# Run POPE evaluation (full dataset)
echo "=========================================="
echo "ECSE POPE Full Evaluation"
echo "Start time: $(date)"
echo "=========================================="

# Adversarial split
python -m hulluedit.eval.pope_eval \
    --config configs/ecse_pope_llava.yaml \
    --split adversarial \
    --output outputs/pope_adversarial.json

echo ""
echo "=========================================="
echo "Evaluation completed: $(date)"
echo "=========================================="
