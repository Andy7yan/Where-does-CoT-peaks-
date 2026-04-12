#!/bin/bash

set -euo pipefail

: "${RUN_NAME:=generate-remaining-$(date +%m%d_%H%M%S)}"
: "${CONFIG_PATH:=configs/stage1.yaml}"
: "${START_IDX:=200}"
: "${END_IDX:=}"
: "${QUESTIONS_PER_SHARD:=160}"
: "${JOB_SCRIPT:=jobs/generate.pbs}"

export RUN_NAME CONFIG_PATH SUBSET_PATH START_IDX END_IDX QUESTIONS_PER_SHARD JOB_SCRIPT

"$(dirname "$0")/submit_generate_shards.sh"
