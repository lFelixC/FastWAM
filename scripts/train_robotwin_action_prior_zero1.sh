#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for RoboTwin FastWAM action-prior training.
#
# Default behavior is a bounded ZeRO-1 smoke run:
#   bash scripts/train_robotwin_action_prior_zero1.sh
#
# Full training:
#   RUN_MODE=train GPUS=8 bash scripts/train_robotwin_action_prior_zero1.sh
#
# Common parameters, all overridable from environment:
#   RUN_MODE            smoke | train. smoke auto-stops after SMOKE_STEPS.
#   GPUS                Processes per node. smoke default: 1; train default: 8.
#   CUDA_VISIBLE_DEVICES GPU list. smoke default: 0; train default: keep current.
#   TASK                Hydra task config. Default: robotwin_uncond_3cam_384_1e-4.
#   MODEL               Hydra model config. Default: fastwam.
#   NUM_FREQ            DCT low-frequency count K. Default: 6.
#   PRIOR_NOISE_SCALE   Noise added to prior source. Default: 0.0.
#   LOSS_WEIGHT         Weight for loss_prior. Default: 1.0.
#   BATCH_SIZE          Per-process micro batch. smoke default: 1; train default: 16.
#   NUM_WORKERS         Dataloader workers. smoke default: 0; train default: 4.
#   MAX_STEPS           Hydra max_steps. smoke uses a large value to avoid final save.
#   SAVE_EVERY          Checkpoint interval. smoke default: 0.
#   EVAL_EVERY          Eval interval. smoke default: 0.
#   LOG_EVERY           Train log interval. smoke default: 1.
#   SMOKE_STEPS         In smoke mode, stop after this logged training step. Default: 2.
#   SMOKE_TIMEOUT_SEC   In smoke mode, fail if SMOKE_STEPS is not reached in time.
#   OUTPUT_ROOT         Parent run directory.
#   RUN_ID              Run directory name. Default: timestamp.
#   WANDB_ENABLED       Hydra wandb.enabled. Default: false.
#   WANDB_NAME          WandB run name.
#   MASTER_PORT         Torch distributed port. Default: 29500.
#
# Extra Hydra overrides can be appended after the script, for example:
#   RUN_MODE=train GPUS=8 bash scripts/train_robotwin_action_prior_zero1.sh \
#     model.action_source.num_freq=8 learning_rate=5e-5

usage() {
  cat <<'EOF'
One-click launcher for RoboTwin FastWAM action-prior training.

Default bounded ZeRO-1 smoke run:
  bash scripts/train_robotwin_action_prior_zero1.sh

Full training:
  RUN_MODE=train GPUS=8 bash scripts/train_robotwin_action_prior_zero1.sh

Common parameters, all overridable from environment:
  RUN_MODE            smoke | train. smoke auto-stops after SMOKE_STEPS.
  GPUS                Processes per node. smoke default: 1; train default: 8.
  CUDA_VISIBLE_DEVICES GPU list. smoke default: 0; train default: keep current.
  TASK                Hydra task config. Default: robotwin_uncond_3cam_384_1e-4.
  MODEL               Hydra model config. Default: fastwam.
  NUM_FREQ            DCT low-frequency count K. Default: 6.
  PRIOR_NOISE_SCALE   Noise added to prior source. Default: 0.0.
  LOSS_WEIGHT         Weight for loss_prior. Default: 1.0.
  BATCH_SIZE          Per-process micro batch. smoke default: 1; train default: 16.
  NUM_WORKERS         Dataloader workers. smoke default: 0; train default: 4.
  MAX_STEPS           Hydra max_steps. smoke uses a large value to avoid final save.
  SAVE_EVERY          Checkpoint interval. smoke default: 0.
  EVAL_EVERY          Eval interval. smoke default: 0.
  LOG_EVERY           Train log interval. smoke default: 1.
  SMOKE_STEPS         In smoke mode, stop after this logged training step. Default: 2.
  SMOKE_TIMEOUT_SEC   In smoke mode, fail if SMOKE_STEPS is not reached in time.
  OUTPUT_ROOT         Parent run directory.
  RUN_ID              Run directory name. Default: timestamp.
  WANDB_ENABLED       Hydra wandb.enabled. Default: false.
  WANDB_NAME          WandB run name.
  MASTER_PORT         Torch distributed port. Default: 29500.

Extra Hydra overrides can be appended after the script, for example:
  RUN_MODE=train GPUS=8 bash scripts/train_robotwin_action_prior_zero1.sh \
    model.action_source.num_freq=8 learning_rate=5e-5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/envs/fastwam/bin/python}"
BASE_LAUNCHER="${REPO_ROOT}/scripts/train_robotwin_zero1.sh"

RUN_MODE="${RUN_MODE:-smoke}"
if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "train" ]]; then
  echo "Error: RUN_MODE must be 'smoke' or 'train', got '${RUN_MODE}'." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${BASE_LAUNCHER}" ]]; then
  echo "Error: base launcher not found or not executable: ${BASE_LAUNCHER}" >&2
  exit 1
fi

TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
MODEL="${MODEL:-fastwam}"
NUM_FREQ="${NUM_FREQ:-6}"
PRIOR_NOISE_SCALE="${PRIOR_NOISE_SCALE:-0.0}"
LOSS_WEIGHT="${LOSS_WEIGHT:-1.0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  GPUS="${GPUS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  MAX_STEPS="${MAX_STEPS:-100000}"
  SAVE_EVERY="${SAVE_EVERY:-0}"
  EVAL_EVERY="${EVAL_EVERY:-0}"
  LOG_EVERY="${LOG_EVERY:-1}"
  SMOKE_STEPS="${SMOKE_STEPS:-2}"
  SMOKE_TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-1200}"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  fi
else
  GPUS="${GPUS:-8}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
  MAX_STEPS="${MAX_STEPS:-}"
  SAVE_EVERY="${SAVE_EVERY:-2500}"
  EVAL_EVERY="${EVAL_EVERY:-500}"
  LOG_EVERY="${LOG_EVERY:-10}"
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)-action-prior-${RUN_MODE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/robotwin_action_prior_zero1}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-robotwin_action_prior_${RUN_MODE}}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"

HYDRA_ARGS=(
  "task=${TASK}"
  "model=${MODEL}"
  "model.action_source.type=low_freq_prior"
  "model.action_source.num_freq=${NUM_FREQ}"
  "model.action_source.prior_noise_scale=${PRIOR_NOISE_SCALE}"
  "model.action_source.loss_weight=${LOSS_WEIGHT}"
  "batch_size=${BATCH_SIZE}"
  "num_workers=${NUM_WORKERS}"
  "save_every=${SAVE_EVERY}"
  "eval_every=${EVAL_EVERY}"
  "log_every=${LOG_EVERY}"
  "wandb.enabled=${WANDB_ENABLED}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  HYDRA_ARGS+=("max_steps=${MAX_STEPS}")
fi
HYDRA_ARGS+=("$@")

export REPO_ROOT
export PYTHON_BIN
export OUTPUT_DIR
export RUN_ID
export WANDB_NAME
export MASTER_PORT

echo "[action_prior] mode=${RUN_MODE} repo=${REPO_ROOT}"
echo "[action_prior] python=${PYTHON_BIN}"
echo "[action_prior] gpus=${GPUS} cuda_visible=${CUDA_VISIBLE_DEVICES:-<unchanged>}"
echo "[action_prior] output_dir=${OUTPUT_DIR}"
printf '[action_prior_args]'
printf ' %q' "${HYDRA_ARGS[@]}"
printf '\n'

if [[ "${RUN_MODE}" == "train" ]]; then
  exec bash "${BASE_LAUNCHER}" "${GPUS}" "${HYDRA_ARGS[@]}"
fi

TRAIN_LOG="${OUTPUT_DIR}/train.log"
: > "${TRAIN_LOG}"

echo "[action_prior] smoke will stop after step=${SMOKE_STEPS} with loss_prior present."
echo "[action_prior] streaming log: ${TRAIN_LOG}"

setsid bash "${BASE_LAUNCHER}" "${GPUS}" "${HYDRA_ARGS[@]}" >> "${TRAIN_LOG}" 2>&1 &
LAUNCH_PID=$!
tail -n +1 -f "${TRAIN_LOG}" &
TAIL_PID=$!

cleanup_tail() {
  kill "${TAIL_PID}" >/dev/null 2>&1 || true
}
trap cleanup_tail EXIT

START_TS="$(date +%s)"
SMOKE_OK=0
while kill -0 "${LAUNCH_PID}" >/dev/null 2>&1; do
  if grep -q "step=${SMOKE_STEPS}/" "${TRAIN_LOG}" && grep -q "loss_prior" "${TRAIN_LOG}"; then
    SMOKE_OK=1
    echo "[action_prior] reached smoke target step=${SMOKE_STEPS}; stopping launcher before final checkpoint."
    kill -INT "-${LAUNCH_PID}" >/dev/null 2>&1 || kill -INT "${LAUNCH_PID}" >/dev/null 2>&1 || true
    break
  fi

  NOW_TS="$(date +%s)"
  if (( NOW_TS - START_TS > SMOKE_TIMEOUT_SEC )); then
    echo "Error: smoke timeout after ${SMOKE_TIMEOUT_SEC}s before step=${SMOKE_STEPS}." >&2
    kill -INT "-${LAUNCH_PID}" >/dev/null 2>&1 || kill -INT "${LAUNCH_PID}" >/dev/null 2>&1 || true
    break
  fi
  sleep 5
done

set +e
wait "${LAUNCH_PID}"
LAUNCH_STATUS=$?
set -e

cleanup_tail

if (( SMOKE_OK == 1 )); then
  echo "[action_prior] smoke passed. Launcher exit status after intentional SIGINT: ${LAUNCH_STATUS}."
  exit 0
fi

echo "Error: smoke failed; launcher exit status=${LAUNCH_STATUS}. See ${TRAIN_LOG}" >&2
exit "${LAUNCH_STATUS}"
