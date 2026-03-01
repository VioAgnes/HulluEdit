cd /path/to/HulluEdit
mkdir -p outputs/pope_llava

# 运行 random
python hulluedit/eval/pope_eval.py \
    --config configs/pope_llava_run.yaml \
    --split random \
    --output outputs/pope_llava/pope_random.json \
    --model-name "LLaVA-1.5"

# 运行 popular
python hulluedit/eval/pope_eval.py \
    --config configs/pope_llava_run.yaml \
    --split popular \
    --output outputs/pope_llava/pope_popular.json \
    --model-name "LLaVA-1.5"

# 运行 adversarial
python hulluedit/eval/pope_eval.py \
    --config configs/pope_llava_run.yaml \
    --split adversarial \
    --output outputs/pope_llava/pope_adversarial.json \
    --model-name "LLaVA-1.5"

# 聚合结果
python hulluedit/eval/aggregate_pope.py \
    --files outputs/pope_llava/pope_random.json outputs/pope_llava/pope_popular.json outputs/pope_llava/pope_adversarial.json \
    --output outputs/pope_llava/pope_all_metrics.json