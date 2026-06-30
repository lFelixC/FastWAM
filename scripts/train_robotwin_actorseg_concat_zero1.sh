#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_robotwin_actorseg_concat_zero1.sh [gpus_per_node] [hydra_overrides...]

Environment overrides:
  REPO_ROOT=/data/test/FastWAM
  PYTHON_BIN=/root/miniconda3/envs/fastwam/bin/python
  ROBOTWIN_BASE_DIR=/data/datasets/robotwin2.0-fastwam/robotwin2.0-fastwam/robotwin2.0
  ROBOTWIN_ACTORSEG_DIR=/data/datasets/robotwin2_actorseg_50tasks_5eps_lerobot/robotwin2.0
  ROBOTWIN_STATS=/data/datasets/robotwin2.0-fastwam/robotwin2.0-fastwam/dataset_stats.json
  TEXT_CACHE_DIR=/data/test/FastWAM/data/text_embeds_cache/robotwin2_fastwam_full_actorseg_10pct
  WAN22_MODEL_ID=/data/checkpoints/dreamzero/Wan2.2-TI2V-5B
  ACTION_DIT_PATH=/data/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  FASTWAM_SWANLAB_ENABLED=true
  FASTWAM_SWANLAB_PROJECT=fastwam-robotwin
  FASTWAM_SWANLAB_NAME=<defaults to WANDB_NAME>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-/data/test/FastWAM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fastwam/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}"
TASK="${TASK:-robotwin_actorseg_concat_3cam_384_1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

ROBOTWIN_BASE_DIR="${ROBOTWIN_BASE_DIR:-/data/datasets/robotwin2.0-fastwam/robotwin2.0-fastwam/robotwin2.0}"
ROBOTWIN_ACTORSEG_DIR="${ROBOTWIN_ACTORSEG_DIR:-/data/datasets/robotwin2_actorseg_50tasks_5eps_lerobot/robotwin2.0}"
ROBOTWIN_STATS="${ROBOTWIN_STATS:-/data/datasets/robotwin2.0-fastwam/robotwin2.0-fastwam/dataset_stats.json}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/data/test/FastWAM/data/text_embeds_cache/robotwin2_fastwam_full_actorseg_10pct}"
WAN22_MODEL_ID="${WAN22_MODEL_ID:-/data/checkpoints/dreamzero/Wan2.2-TI2V-5B}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-${WAN22_MODEL_ID}}"
ACTION_DIT_PATH="${ACTION_DIT_PATH:-/data/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FASTWAM_SWANLAB_ENABLED="${FASTWAM_SWANLAB_ENABLED:-${SWANLAB_ENABLED:-true}}"
FASTWAM_SWANLAB_PROJECT="${FASTWAM_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-fastwam-robotwin}}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if [[ $# -gt 0 && "$(is_integer "${1}" && echo yes || echo no)" == "yes" ]]; then
  NPROC_PER_NODE="${1}"
  shift
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Error: FastWAM repo not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${ROBOTWIN_BASE_DIR}" ]]; then
  echo "Error: ROBOTWIN_BASE_DIR is missing: ${ROBOTWIN_BASE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${ROBOTWIN_ACTORSEG_DIR}" ]]; then
  echo "Error: ROBOTWIN_ACTORSEG_DIR is missing: ${ROBOTWIN_ACTORSEG_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_STATS}" ]]; then
  echo "Error: ROBOTWIN_STATS is missing: ${ROBOTWIN_STATS}" >&2
  exit 1
fi
if [[ ! -d "${WAN22_MODEL_ID}" ]]; then
  echo "Error: WAN22_MODEL_ID is missing: ${WAN22_MODEL_ID}" >&2
  exit 1
fi
if [[ ! -f "${ACTION_DIT_PATH}" ]]; then
  echo "Error: ACTION_DIT_PATH is missing: ${ACTION_DIT_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${TEXT_CACHE_DIR}"

TOTAL_PROCESSES=$((NNODES * NPROC_PER_NODE))
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${TASK}/${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-${TASK}}"
FASTWAM_SWANLAB_NAME="${FASTWAM_SWANLAB_NAME:-${SWANLAB_NAME:-${WANDB_NAME}}}"

# SwanLab's own settings parser consumes SWANLAB_PROJECT/SWANLAB_NAME as
# structured SDK settings. Use FastWAM-prefixed variables for launcher
# configuration and keep only SWANLAB_API_KEY for login inside Python.
unset SWANLAB_ENABLED SWANLAB_PROJECT SWANLAB_NAME

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${WAN22_MODEL_ID}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -x "${ACCELERATE_BIN}" ]]; then
  LAUNCHER=("${ACCELERATE_BIN}" launch)
else
  LAUNCHER=("${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch)
fi

LAUNCH_CMD=(
  "${LAUNCHER[@]}"
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --num_processes "${TOTAL_PROCESSES}"
  --num_machines "${NNODES}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  scripts/train.py
  "task=${TASK}"
  "output_dir=${OUTPUT_DIR}"
  "wandb.name=${WANDB_NAME}"
  "swanlab.enabled=${FASTWAM_SWANLAB_ENABLED}"
  "swanlab.project=${FASTWAM_SWANLAB_PROJECT}"
  "swanlab.name=${FASTWAM_SWANLAB_NAME}"
  "data.train.datasets.0.dataset.dataset_dirs=[${ROBOTWIN_BASE_DIR}]"
  "data.train.datasets.1.dataset.dataset_dirs=[${ROBOTWIN_ACTORSEG_DIR}]"
  "data.val.dataset_dirs=[${ROBOTWIN_BASE_DIR}]"
  "data.train.pretrained_norm_stats=${ROBOTWIN_STATS}"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
  "model.model_id=${WAN22_MODEL_ID}"
  "model.tokenizer_model_id=${TOKENIZER_MODEL_ID}"
  "model.redirect_common_files=false"
  "model.action_dit_pretrained_path=${ACTION_DIT_PATH}"
  "$@"
)

printf '[launch_cmd]'
printf ' %q' "${LAUNCH_CMD[@]}"
printf '\n'

exec "${LAUNCH_CMD[@]}"
