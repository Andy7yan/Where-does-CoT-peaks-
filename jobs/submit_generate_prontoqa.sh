#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

: "${RUN_NAME:=prontoqa-paper-$(date +%m%d_%H%M%S)}"
: "${QUESTIONS_PER_SHARD:=250}"

CONFIG_PATH="configs/stage1_prontoqa.yaml" \
JOB_SCRIPT="jobs/generate_prontoqa.pbs" \
QUESTIONS_PER_SHARD="${QUESTIONS_PER_SHARD}" \
RUN_NAME="${RUN_NAME}" \
bash jobs/submit_generate_shards.sh
