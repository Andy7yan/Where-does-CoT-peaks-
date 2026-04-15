#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set +u
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
  set -u
fi

: "${RUN_NAME:=generate-$(date +%m%d_%H%M%S)}"
: "${START_IDX:=0}"
: "${END_IDX:=}"
: "${QUESTIONS_PER_SHARD:=300}"
: "${JOB_SCRIPT:=jobs/generate.pbs}"
: "${CONFIG_PATH:=configs/stage1.yaml}"
: "${SOURCE:=huggingface}"
: "${LOCAL_PATH:=}"
: "${CACHE_DIR:=${HF_DATASETS_CACHE:-}}"

if (( START_IDX < 0 )); then
  echo "START_IDX must be non-negative"
  exit 1
fi

if (( QUESTIONS_PER_SHARD <= 0 )); then
  echo "QUESTIONS_PER_SHARD must be positive"
  exit 1
fi

python scripts/check_generation_preflight.py --config "${CONFIG_PATH}"

dataset_size_cmd=(
  python
  scripts/print_dataset_size.py
  --config "${CONFIG_PATH}"
  --source "${SOURCE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  dataset_size_cmd+=(--cache-dir "${CACHE_DIR}")
fi

if [[ -n "${LOCAL_PATH}" ]]; then
  dataset_size_cmd+=(--local-path "${LOCAL_PATH}")
fi

dataset_size="$("${dataset_size_cmd[@]}")"
if ! [[ "${dataset_size}" =~ ^[0-9]+$ ]]; then
  echo "failed to resolve dataset size: ${dataset_size}"
  exit 1
fi

if [[ -z "${END_IDX}" ]]; then
  END_IDX=${dataset_size}
fi

if (( END_IDX <= START_IDX )); then
  echo "END_IDX must be greater than START_IDX"
  exit 1
fi

if (( END_IDX > dataset_size )); then
  echo "END_IDX=${END_IDX} exceeds dataset size ${dataset_size}"
  exit 1
fi

total_questions=$(( END_IDX - START_IDX ))
num_shards=$(( (total_questions + QUESTIONS_PER_SHARD - 1) / QUESTIONS_PER_SHARD ))
start_idx=${START_IDX}

echo "run_name=${RUN_NAME}"
echo "config_path=${CONFIG_PATH}"
echo "source=${SOURCE}"
echo "local_path=${LOCAL_PATH}"
echo "dataset_size=${dataset_size}"
echo "start_idx=${START_IDX}"
echo "end_idx=${END_IDX}"
echo "questions_per_shard=${QUESTIONS_PER_SHARD}"
echo "num_shards=${num_shards}"
echo "total_questions=${total_questions}"
echo "job_script=${JOB_SCRIPT}"

for (( shard_index=0; shard_index<num_shards; shard_index++ )); do
  end_idx=$(( start_idx + QUESTIONS_PER_SHARD ))
  if (( end_idx > END_IDX )); then
    end_idx=${END_IDX}
  fi
  shard_id=$(printf 'q%04d_%04d' "${start_idx}" "${end_idx}")

  echo "submitting shard=$(( shard_index + 1 ))/${num_shards} start=${start_idx} end=${end_idx} shard_id=${shard_id}"
  qsub -v RUN_NAME="${RUN_NAME}",CONFIG_PATH="${CONFIG_PATH}",SOURCE="${SOURCE}",LOCAL_PATH="${LOCAL_PATH}",START_IDX="${start_idx}",END_IDX="${end_idx}",SHARD_ID="${shard_id}" "${JOB_SCRIPT}"

  start_idx=${end_idx}
done

echo "aggregate_after_finish=python scripts/run_aggregation.py --stage e --run-dir /srv/scratch/${USER}/peak-CoT/runs/${RUN_NAME} --config ${CONFIG_PATH}"
