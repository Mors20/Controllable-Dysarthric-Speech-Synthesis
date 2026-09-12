#!/usr/bin/env bash
set -euo pipefail

: "${TORGO_ROOT:?set TORGO_ROOT to the extracted TORGO directory}"
: "${MODEL_DIR:?set MODEL_DIR to the IndexTTS-1.5 pretrained model directory}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/data}"
CONVERTED_ROOT="${CONVERTED_ROOT:-${WORK_ROOT}/converted}"
PREPARED_ROOT="${PREPARED_ROOT:-${WORK_ROOT}/prepared}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}/third_party/seed-vc"
"${PYTHON_BIN}" generate_data.py \
  --mode both \
  --torgo_root "${TORGO_ROOT}" \
  --output "${CONVERTED_ROOT}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/prepare_torgo.py \
  --torgo_root "${TORGO_ROOT}" \
  --converted_root "${CONVERTED_ROOT}" \
  --out_dir "${PREPARED_ROOT}" \
  --finetune_dir "${MODEL_DIR}" \
  --config configs/controllable_dysarthric_speech_synthesis.yaml

"${PYTHON_BIN}" train.py \
  --config configs/controllable_dysarthric_speech_synthesis.yaml \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${PREPARED_ROOT}" \
  --embedding-dir "${PREPARED_ROOT}/pathology_embedding"
