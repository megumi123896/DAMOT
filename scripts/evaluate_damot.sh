#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
VAL_ROOT="${VAL_ROOT:-${AUTODL_ROOT}/data/VisDrone2019-MOT-val}"
TEST_DEV_ROOT="${TEST_DEV_ROOT:-${AUTODL_ROOT}/data/VisDrone2019-MOT-test-dev}"
SPLIT="${1:-val}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/model/best_ckpt.pth.tar}"
REID_WEIGHTS="${REID_WEIGHTS:-${REPO_ROOT}/pretrained/veriwild_bot_R50-ibn.pth}"
DEVICE="${DEVICE:-0}"
REID_BATCH_SIZE="${REID_BATCH_SIZE:-128}"

if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test-dev" ]]; then
  echo "Split must be val or test-dev, got: ${SPLIT}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" tools/prepare_visdrone.py \
  --val-root "${VAL_ROOT}" \
  --test-dev-root "${TEST_DEV_ROOT}" \
  --output-root "${REPO_ROOT}/datasets/visdrone" \
  --splits "${SPLIT}"

"${PYTHON_BIN}" tools/eval_damot.py \
  -c "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --val-root "${VAL_ROOT}" \
  --test-dev-root "${TEST_DEV_ROOT}" \
  --reid-weights "${REID_WEIGHTS}" \
  --device "${DEVICE}" \
  --reid-batch-size "${REID_BATCH_SIZE}" \
  --fp16
