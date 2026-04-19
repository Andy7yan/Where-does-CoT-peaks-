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

: "${RUN_NAME:=per-question-$(date +%m%d_%H%M%S)}"
: "${CONFIG_PATH:=configs/stage1_per_question.yaml}"
: "${SOURCE_RUN:=generate-rerun-0416_165235}"
: "${PROMPTS_DIR:=prompts/per_question}"
: "${SOURCE:=huggingface}"
: "${LOCAL_PATH:=}"
: "${CACHE_DIR:=${HF_DATASETS_CACHE:-}}"
: "${QUESTIONS_PER_SHARD:=}"
: "${TARGET_TRACES_PER_SHARD:=7200}"
: "${START_IDX:=0}"
: "${END_IDX:=}"
: "${JOB_SCRIPT:=jobs/generate_per_question.pbs}"

# Backward-compatible fallback for older shell snippets.
if [[ -n "${SOURCE_V6_RUN:-}" && "${SOURCE_RUN}" == "generate-rerun-0416_165235" ]]; then
  SOURCE_RUN="${SOURCE_V6_RUN}"
fi

if (( START_IDX < 0 )); then
  echo "START_IDX must be non-negative"
  exit 1
fi

if [[ -n "${QUESTIONS_PER_SHARD}" ]] && (( QUESTIONS_PER_SHARD <= 0 )); then
  echo "QUESTIONS_PER_SHARD must be positive"
  exit 1
fi

if (( TARGET_TRACES_PER_SHARD <= 0 )); then
  echo "TARGET_TRACES_PER_SHARD must be positive"
  exit 1
fi

: "${SCRATCH:?SCRATCH must be available from .env for per-question submission}"
RUN_DIR="${SCRATCH}/runs/${RUN_NAME}"
QUESTION_MANIFEST_PATH="${RUN_DIR}/per_question_manifest.jsonl"

python scripts/check_generation_preflight.py --config "${CONFIG_PATH}" --prompts-dir "${PROMPTS_DIR}"

build_manifest_cmd=(
  python
  scripts/build_per_question_manifest.py
  --config "${CONFIG_PATH}"
  --source-run "${SOURCE_RUN}"
  --output-dir "${RUN_DIR}"
  --source "${SOURCE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  build_manifest_cmd+=(--cache-dir "${CACHE_DIR}")
fi

if [[ -n "${LOCAL_PATH}" ]]; then
  build_manifest_cmd+=(--local-path "${LOCAL_PATH}")
fi

"${build_manifest_cmd[@]}"

if [[ ! -f "${QUESTION_MANIFEST_PATH}" ]]; then
  echo "missing per-question manifest at ${QUESTION_MANIFEST_PATH}"
  exit 1
fi

manifest_size=$(wc -l < "${QUESTION_MANIFEST_PATH}")
manifest_size="${manifest_size//[[:space:]]/}"
if ! [[ "${manifest_size}" =~ ^[0-9]+$ ]]; then
  echo "failed to resolve per-question manifest size: ${manifest_size}"
  exit 1
fi

if [[ -z "${END_IDX}" ]]; then
  END_IDX=${manifest_size}
fi

if (( END_IDX <= START_IDX )); then
  echo "END_IDX must be greater than START_IDX"
  exit 1
fi

if (( END_IDX > manifest_size )); then
  echo "END_IDX=${END_IDX} exceeds per-question manifest size ${manifest_size}"
  exit 1
fi

total_questions=$(( END_IDX - START_IDX ))
SHARD_PLAN_PATH="${RUN_DIR}/per_question_shards.jsonl"
shard_plan_cmd=(
  python
  scripts/build_per_question_shard_plan.py
  --question-manifest "${QUESTION_MANIFEST_PATH}"
  --target-traces-per-shard "${TARGET_TRACES_PER_SHARD}"
  --output-path "${SHARD_PLAN_PATH}"
  --start-idx "${START_IDX}"
  --format tsv
)

if [[ -n "${QUESTIONS_PER_SHARD}" ]]; then
  shard_plan_cmd+=(--questions-per-shard "${QUESTIONS_PER_SHARD}")
fi

if [[ -n "${END_IDX}" ]]; then
  shard_plan_cmd+=(--end-idx "${END_IDX}")
fi

mapfile -t shard_rows < <("${shard_plan_cmd[@]}")
if (( ${#shard_rows[@]} == 0 )); then
  echo "failed to build a per-question shard plan"
  exit 1
fi

num_shards=${#shard_rows[@]}

echo "run_name=${RUN_NAME}"
echo "config_path=${CONFIG_PATH}"
echo "source_run=${SOURCE_RUN}"
echo "prompt_dir=${PROMPTS_DIR}"
echo "question_manifest_path=${QUESTION_MANIFEST_PATH}"
echo "shard_plan_path=${SHARD_PLAN_PATH}"
echo "manifest_size=${manifest_size}"
echo "start_idx=${START_IDX}"
echo "end_idx=${END_IDX}"
echo "questions_per_shard=${QUESTIONS_PER_SHARD:-auto}"
echo "target_traces_per_shard=${TARGET_TRACES_PER_SHARD}"
echo "num_shards=${num_shards}"
echo "total_questions=${total_questions}"
echo "job_script=${JOB_SCRIPT}"

for shard_row in "${shard_rows[@]}"; do
  IFS=$'\t' read -r shard_index start_idx end_idx question_count shard_traces <<< "${shard_row}"
  shard_id=$(printf 'q%04d_%04d' "${start_idx}" "${end_idx}")

  echo "submitting shard=$(( shard_index + 1 ))/${num_shards} start=${start_idx} end=${end_idx} questions=${question_count} target_traces=${shard_traces} shard_id=${shard_id}"
  qsub -v RUN_NAME="${RUN_NAME}",CONFIG_PATH="${CONFIG_PATH}",PROMPTS_DIR="${PROMPTS_DIR}",QUESTION_MANIFEST_PATH="${QUESTION_MANIFEST_PATH}",START_IDX="${start_idx}",END_IDX="${end_idx}",SHARD_ID="${shard_id}" "${JOB_SCRIPT}"
done
