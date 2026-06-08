#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${OPENGS_PYTHON:-/home/leizongru/miniconda3/envs/opengs-slam/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-$REPO_DIR/configs/mono/railway/railway.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/leizongru/lzr_ws/railway_data}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65535}"
CLEAN_USER_OUTPUT="${CLEAN_USER_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if ! ulimit -n "$OPEN_FILE_LIMIT" 2>/dev/null; then
  echo "WARN: unable to set open file limit to $OPEN_FILE_LIMIT; current limit is $(ulimit -n)" >&2
fi

DEFAULT_SCENES=(
  scene_05_train
  scene_11_train
  scene_13_train
  scene_14_train
  scene_16_train
  scene_17_train
  scene_19_train
)

mkdir -p "$LOG_DIR"
TMP_CONFIG_DIR="$LOG_DIR/${RUN_TAG}_configs"
mkdir -p "$TMP_CONFIG_DIR"
SUMMARY_FILE="$LOG_DIR/${RUN_TAG}_summary.tsv"
QUEUE_FILE="$LOG_DIR/${RUN_TAG}_queue.txt"
QUEUE_LOCK="$LOG_DIR/${RUN_TAG}_queue.lock"
SUMMARY_LOCK="$LOG_DIR/${RUN_TAG}_summary.lock"
FAILED_FILE="$LOG_DIR/${RUN_TAG}_failed.txt"
WORKER_PIDS=()

cleanup_batch() {
  trap - INT TERM
  if [[ "${#WORKER_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping workers: ${WORKER_PIDS[*]}" >&2
    kill -TERM "${WORKER_PIDS[@]}" 2>/dev/null || true
    wait "${WORKER_PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}

normalize_list() {
  tr ',' ' ' <<< "${1:-}" | xargs -n1
}

detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    normalize_list "$GPU_IDS"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found and GPU_IDS is not set." >&2
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, -v max_mb="$GPU_MAX_USED_MB" '{
      gpu=$1; mem=$2;
      gsub(/^[ \t]+|[ \t]+$/, "", gpu);
      gsub(/^[ \t]+|[ \t]+$/, "", mem);
      if (mem + 0 <= max_mb) print gpu;
    }'
}

acquire_lock() {
  local lock_dir="$1"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 0.1
  done
}

release_lock() {
  rmdir "$1"
}

pick_scene() {
  local scene=""
  acquire_lock "$QUEUE_LOCK"
  if [[ -s "$QUEUE_FILE" ]]; then
    scene="$(head -n 1 "$QUEUE_FILE")"
    tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
  fi
  release_lock "$QUEUE_LOCK"
  printf '%s' "$scene"
}

append_summary() {
  acquire_lock "$SUMMARY_LOCK"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$SUMMARY_FILE"
  release_lock "$SUMMARY_LOCK"
}

clean_user_output() {
  local scene="$1"
  local scene_output="$REPO_DIR/output/$scene"
  [[ "$CLEAN_USER_OUTPUT" == "1" ]] || return 0
  rm -rf "$scene_output/renders" "$scene_output/poses" "$scene_output/metrics" "$scene_output/config.yaml"
}

make_scene_config() {
  local scene="$1"
  local config_path="$2"
  DATA_ROOT_PATH="$DATA_ROOT" SCENE="$scene" CONFIG_OUT="$config_path" BASE_CONFIG_PATH="$BASE_CONFIG" REPO_DIR_PATH="$REPO_DIR" "$PYTHON_BIN" - <<'PYCONFIG'
import os
from pathlib import Path
import yaml

base_config = Path(os.environ["BASE_CONFIG_PATH"])
repo = Path(os.environ["REPO_DIR_PATH"])
scene = os.environ["SCENE"]
config_out = Path(os.environ["CONFIG_OUT"])

with base_config.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset = cfg.setdefault("Dataset", {})
results = cfg.setdefault("Results", {})

dataset_root = Path(
    os.environ.get("DATA_ROOT_PATH")
    or dataset.get("dataset_root")
    or "/home/leizongru/lzr_ws/railway_data"
)

dataset["dataset_root"] = str(dataset_root)
dataset["scene"] = scene
dataset["dataset_path"] = str(dataset_root / scene)
dataset["device"] = "cuda:0"
results["save_dir"] = str(repo / "output")
results["clean_export"] = True

with config_out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYCONFIG
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local scene_config="$TMP_CONFIG_DIR/${scene}.yaml"
  local log_file="$LOG_DIR/${RUN_TAG}_${scene}_gpu${gpu}.log"
  local status="OK"
  local exit_code=0
  local start_ts
  local end_ts
  local scene_pid=""

  stop_scene() {
    trap - INT TERM
    if [[ -n "$scene_pid" ]] && kill -0 "$scene_pid" 2>/dev/null; then
      kill -TERM "$scene_pid" 2>/dev/null || true
      wait "$scene_pid" 2>/dev/null || true
    fi
    exit 130
  }

  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "run_tag: $RUN_TAG"
    echo "scene: $scene"
    echo "gpu: $gpu"
    echo "config: $scene_config"
    echo "start: $start_ts"
    echo "repo: $REPO_DIR"
    echo "python: $PYTHON_BIN"
    echo "data_root: $DATA_ROOT"
    echo "open_file_limit: $(ulimit -n)"
    echo "pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF:-}"
    echo
    echo "command:"
    echo "CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN slam.py --config $scene_config $EXTRA_ARGS"
    echo
  } > "$log_file"

  set +e
  make_scene_config "$scene" "$scene_config" >> "$log_file" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 && "$DRY_RUN" != "1" ]]; then
    clean_user_output "$scene" >> "$log_file" 2>&1
    exit_code=$?
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN=1, skip execution." >> "$log_file"
    else
      trap stop_scene INT TERM
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" slam.py --config "$scene_config" $EXTRA_ARGS >> "$log_file" 2>&1 &
      scene_pid=$!
      wait "$scene_pid"
      exit_code=$?
      scene_pid=""
      trap - INT TERM
    fi
  fi
  set -e

  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$exit_code" -ne 0 ]]; then
    status="FAIL"
    echo "$scene" >> "$FAILED_FILE"
  fi
  {
    echo
    echo "end: $end_ts"
    echo "exit_code: $exit_code"
    echo "status: $status"
  } >> "$log_file"

  append_summary "$scene" "$gpu" "$status" "$exit_code" "$start_ts" "$end_ts"
}

worker() {
  local gpu="$1"
  local scene
  while true; do
    scene="$(pick_scene)"
    [[ -n "$scene" ]] || break
    run_scene "$gpu" "$scene"
  done
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

mapfile -t GPU_LIST < <(detect_gpus)
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: no available GPU found. Set GPU_IDS manually, e.g. GPU_IDS=2,3 $0" >&2
  exit 1
fi

if [[ -n "${SCENES:-}" ]]; then
  mapfile -t SCENE_LIST < <(normalize_list "$SCENES")
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

for scene in "${SCENE_LIST[@]}"; do
  if [[ ! -d "$DATA_ROOT/$scene" ]]; then
    echo "ERROR: scene directory not found: $DATA_ROOT/$scene" >&2
    exit 1
  fi
done

printf 'scene\tgpu\tstatus\texit_code\tstart\tend\n' > "$SUMMARY_FILE"
printf '%s\n' "${SCENE_LIST[@]}" > "$QUEUE_FILE"
: > "$FAILED_FILE"

echo "Run tag: $RUN_TAG"
echo "Scenes: ${SCENE_LIST[*]}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Logs: $LOG_DIR"
echo "Summary: $SUMMARY_FILE"

trap cleanup_batch INT TERM
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" &
  WORKER_PIDS+=("$!")
done

wait "${WORKER_PIDS[@]}"

echo
echo "Batch summary:"
cat "$SUMMARY_FILE"

if [[ -s "$FAILED_FILE" ]]; then
  echo
  echo "Failed scenes:"
  cat "$FAILED_FILE"
  exit 1
fi

echo
echo "All scenes finished successfully."
