#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_robotwin_zero1.sh [gpus_per_node] [hydra_overrides...]

Examples:
  # Single node, 8 GPUs
  bash scripts/train_robotwin_zero1.sh 8

  # Multi-node, run one command on each node with the same MASTER_ADDR/MASTER_PORT.
  NNODES=2 NODE_RANK=0 MASTER_ADDR=172.xx.xx.xx MASTER_PORT=29442 bash scripts/train_robotwin_zero1.sh 8
  NNODES=2 NODE_RANK=1 MASTER_ADDR=172.xx.xx.xx MASTER_PORT=29442 bash scripts/train_robotwin_zero1.sh 8

  # Train another RoboTwin task config.
  bash scripts/train_robotwin_zero1.sh 8 task=robotwin_joint_3cam_384_1e-4

Environment defaults:
  REPO_ROOT=/2023133163/liuf/FastWAM
  PYTHON_BIN=/opt/fastwam/bin/python
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR=127.0.0.1
  MASTER_PORT=29500
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-/2023133163/liuf/FastWAM}"
PYTHON_BIN="${PYTHON_BIN:-/opt/fastwam/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}"

DEFAULT_TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if [[ $# -gt 0 && "$(is_integer "${1}" && echo yes || echo no)" == "yes" ]]; then
  NPROC_PER_NODE="${1}"
  shift
fi

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}" || ! is_integer "${MAIN_PROCESS_PORT}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), NNODES (${NUM_MACHINES}), NODE_RANK (${MACHINE_RANK}), and MASTER_PORT (${MAIN_PROCESS_PORT}) must be integers." >&2
  exit 1
fi

RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

if (( NUM_MACHINES < 1 || NPROC_PER_NODE < 1 )); then
  echo "Error: NNODES and gpus_per_node must be >= 1." >&2
  exit 1
fi

if (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: NODE_RANK (${MACHINE_RANK}) must be in [0, NNODES)." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Error: FastWAM repo not found: ${REPO_ROOT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

require_file() {
  if [[ ! -f "${1}" ]]; then
    echo "Error: required file is missing: ${1}" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "${1}" ]]; then
    echo "Error: required directory is missing: ${1}" >&2
    exit 1
  fi
}

require_file "scripts/train.py"
require_file "scripts/accelerate_configs/accelerate_zero1_ds.yaml"
require_file "scripts/ds_configs/ds_zero1_config.json"
require_dir "data/robotwin2.0/robotwin2.0"
require_file "data/robotwin2.0/dataset_stats.json"
require_file "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
require_dir "checkpoints/Wan-AI/Wan2.2-TI2V-5B"
require_dir "checkpoints/Wan-AI/Wan2.1-T2V-1.3B"

has_task_override=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == task=* ]]; then
    has_task_override=true
    break
  fi
done

if [[ "${has_task_override}" == "false" ]]; then
  EXTRA_ARGS=("task=${DEFAULT_TASK}" "${EXTRA_ARGS[@]}")
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

require_file "configs/task/${TASK_BASENAME}.yaml"

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      "${PYTHON_BIN}" - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

TOTAL_PROCESSES=$((NUM_MACHINES * NPROC_PER_NODE))
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${TASK_BASENAME}/${RUN_ID}}"
WANDB_NAME="${WANDB_NAME:-${TASK_BASENAME}}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
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
  --num_machines "${NUM_MACHINES}"
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  scripts/train.py
  "output_dir=${OUTPUT_DIR}"
  "wandb.name=${WANDB_NAME}"
  "${EXTRA_ARGS[@]}"
)

echo "[launch] repo=${REPO_ROOT}"
echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} total_processes=${TOTAL_PROCESSES} machine_rank=${MACHINE_RANK}"
echo "[launch] master=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID} output_dir=${OUTPUT_DIR}"
printf '[launch_cmd]'
printf ' %q' "${LAUNCH_CMD[@]}"
printf '\n'

exec "${LAUNCH_CMD[@]}"
