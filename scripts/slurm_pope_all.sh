#!/bin/bash
#SBATCH -J ecse-pope-all
#SBATCH -o logs/pope_all_%j.out
#SBATCH -e logs/pope_all_%j.err
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

# Create directories
mkdir -p logs outputs

echo "=========================================="
echo "ECSE POPE Full Evaluation (all 3 splits)"
echo "Start time: $(date)"
echo "=========================================="

for SPLIT in random popular adversarial; do
  echo "[POPE] Split: $SPLIT"
  python -m hulluedit.eval.pope_eval \
    --config configs/ecse_pope_llava.yaml \
    --split $SPLIT \
    --output outputs/pope_${SPLIT}.json
done

echo ""
echo "[POPE] Aggregating metrics"
python -m hulluedit.eval.aggregate_pope \
  --files outputs/pope_random.json outputs/pope_popular.json outputs/pope_adversarial.json \
  --output outputs/pope_all_metrics.json

echo ""
echo "=========================================="
echo "Evaluation completed: $(date)"
echo "=========================================="
