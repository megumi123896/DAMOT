#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
TRAIN_ROOT="${TRAIN_ROOT:-${AUTODL_ROOT}/data/VisDrone2019-MOT-train}"
VAL_ROOT="${VAL_ROOT:-${AUTODL_ROOT}/data/VisDrone2019-MOT-val}"
TEST_DEV_ROOT="${TEST_DEV_ROOT:-${AUTODL_ROOT}/data/VisDrone2019-MOT-test-dev}"
DETECTOR_PRETRAINED="${DETECTOR_PRETRAINED:-${REPO_ROOT}/pretrained/yolox_x.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICES="${DEVICES:-1}"
RESUME="${RESUME:-0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" tools/prepare_visdrone.py \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --test-dev-root "${TEST_DEV_ROOT}" \
  --output-root "${REPO_ROOT}/datasets/visdrone" \
  --splits train val test-dev

train_args=(
  tools/train.py
  -f exps/example/mot/damot_visdrone.py
  -d "${DEVICES}"
  -b "${BATCH_SIZE}"
  --fp16
)

if [[ "${RESUME}" == "1" ]]; then
  train_args+=(--resume)
else
  if [[ ! -f "${DETECTOR_PRETRAINED}" ]]; then
    echo "Detector pretrained checkpoint not found: ${DETECTOR_PRETRAINED}" >&2
    exit 1
  fi
  train_args+=(-c "${DETECTOR_PRETRAINED}")
fi

"${PYTHON_BIN}" "${train_args[@]}"
