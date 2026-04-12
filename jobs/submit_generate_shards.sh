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
: "${QUESTIONS_PER_SHARD:=160}"
: "${SUBSET_PATH:=${SCRATCH:-${PROJECT_ROOT}}/data/eval_subset_full.jsonl}"
: "${JOB_SCRIPT:=jobs/generate.pbs}"
: "${CONFIG_PATH:=configs/stage1.yaml}"

if (( START_IDX < 0 )); then
  echo "START_IDX must be non-negative"
  exit 1
fi

if (( QUESTIONS_PER_SHARD <= 0 )); then
  echo "QUESTIONS_PER_SHARD must be positive"
  exit 1
fi

if [[ ! -f "${SUBSET_PATH}" ]]; then
  project_subset_path="${PROJECT_ROOT}/data/eval_subset_full.jsonl"
  if [[ -f "${project_subset_path}" ]]; then
    SUBSET_PATH="${project_subset_path}"
  else
    echo "subset path not found: ${SUBSET_PATH}"
    echo "also checked project path: ${project_subset_path}"
    exit 1
  fi
fi

subset_size=$(wc -l < "${SUBSET_PATH}")
if [[ -z "${END_IDX}" ]]; then
  END_IDX=${subset_size}
fi

if (( END_IDX <= START_IDX )); then
  echo "END_IDX must be greater than START_IDX"
  exit 1
fi

if (( END_IDX > subset_size )); then
  echo "END_IDX=${END_IDX} exceeds subset size ${subset_size}"
  exit 1
fi

total_questions=$(( END_IDX - START_IDX ))
num_shards=$(( (total_questions + QUESTIONS_PER_SHARD - 1) / QUESTIONS_PER_SHARD ))
start_idx=${START_IDX}

echo "run_name=${RUN_NAME}"
echo "subset_path=${SUBSET_PATH}"
echo "config_path=${CONFIG_PATH}"
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
  qsub -v RUN_NAME="${RUN_NAME}",CONFIG_PATH="${CONFIG_PATH}",SUBSET_PATH="${SUBSET_PATH}",START_IDX="${start_idx}",END_IDX="${end_idx}",SHARD_ID="${shard_id}" "${JOB_SCRIPT}"

  start_idx=${end_idx}
done

echo "aggregate_after_finish=python scripts/run_aggregation.py --stage e --run-dir /srv/scratch/${USER}/peak-CoT/runs/${RUN_NAME} --config ${CONFIG_PATH}"
