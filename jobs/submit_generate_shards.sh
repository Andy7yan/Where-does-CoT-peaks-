#!/bin/bash

set -euo pipefail

: "${RUN_NAME:=generate-$(date +%m%d_%H%M%S)}"
: "${NUM_SHARDS:=5}"
: "${TOTAL_QUESTIONS:=200}"
: "${JOB_SCRIPT:=jobs/generate.pbs}"

if (( NUM_SHARDS <= 0 )); then
  echo "NUM_SHARDS must be positive"
  exit 1
fi

if (( TOTAL_QUESTIONS <= 0 )); then
  echo "TOTAL_QUESTIONS must be positive"
  exit 1
fi

base_size=$(( TOTAL_QUESTIONS / NUM_SHARDS ))
remainder=$(( TOTAL_QUESTIONS % NUM_SHARDS ))
start_idx=0

echo "run_name=${RUN_NAME}"
echo "num_shards=${NUM_SHARDS}"
echo "total_questions=${TOTAL_QUESTIONS}"
echo "job_script=${JOB_SCRIPT}"

for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
  shard_size=${base_size}
  if (( shard_index < remainder )); then
    shard_size=$(( shard_size + 1 ))
  fi
  end_idx=$(( start_idx + shard_size ))
  shard_id=$(printf 'q%04d_%04d' "${start_idx}" "${end_idx}")

  echo "submitting shard=$(( shard_index + 1 ))/${NUM_SHARDS} start=${start_idx} end=${end_idx} shard_id=${shard_id}"
  qsub -v RUN_NAME="${RUN_NAME}",START_IDX="${start_idx}",END_IDX="${end_idx}",SHARD_ID="${shard_id}" "${JOB_SCRIPT}"

  start_idx=${end_idx}
done

echo "aggregate_after_finish=python scripts/run_aggregation.py --stage e --run-dir /srv/scratch/${USER}/peak-CoT/runs/${RUN_NAME} --config configs/stage1.yaml"
