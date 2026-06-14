#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-hulluedit}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/path/to/llava-v1.5-7b}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/configs/chair_llava15_7b_nullu.yaml}"
RUN_NAME="${RUN_NAME:-llava15_7b_hulluedit_nullu}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/chair}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-${PROJECT_ROOT}/DATA/nltk_data}"

TMP_CONFIG="${OUTPUT_DIR}/.${RUN_NAME}.yaml"
python - "${CONFIG_TEMPLATE}" "${TMP_CONFIG}" "${MODEL_PATH}" "${PROJECT_ROOT}/DATA" "${OUTPUT_DIR}" <<'PY'
import sys
import yaml

src, dst, model_path, coco_root, output_dir = sys.argv[1:6]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["model_name"] = model_path
cfg["coco_root"] = coco_root
cfg["output_dir"] = output_dir
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

CAP_JSONL="${OUTPUT_DIR}/chair_captions_${RUN_NAME}.jsonl"
CHAIR_JSON="${OUTPUT_DIR}/chair_captions_${RUN_NAME}_chair_result.json"
CHAIR_CACHE="${CHAIR_CACHE:-${OUTPUT_DIR}/chair_${RUN_NAME}.pkl}"
CHAIR_ARGS=()
if [ -n "${CHAIR_CODE_DIR:-}" ]; then
  CHAIR_ARGS+=(--chair-code-dir "${CHAIR_CODE_DIR}")
fi

rm -f "${CAP_JSONL}" "${CHAIR_JSON}" "${CHAIR_CACHE}"

python -m hulluedit.eval.generate_chair_captions \
  --config "${TMP_CONFIG}" \
  --output-file "${CAP_JSONL}" \
  "$@"

python -m hulluedit.eval.eval_chair \
  --input "${CAP_JSONL}" \
  --coco-annotations "${PROJECT_ROOT}/DATA/annotations" \
  --output "${CHAIR_JSON}" \
  --cache "${CHAIR_CACHE}" \
  "${CHAIR_ARGS[@]}"

echo "Captions: ${CAP_JSONL}"
echo "CHAIR: ${CHAIR_JSON}"
