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

: "${RUN_NAME:?RUN_NAME must point to the existing per-question run to repair}"
: "${CONFIG_PATH:=configs/stage1_per_question.yaml}"
: "${PROMPTS_DIR:=prompts/per_question}"
: "${TARGET_TRACES_PER_SHARD:=7200}"
: "${QUESTIONS_PER_SHARD:=}"
: "${ALLOW_APPEND_UNSAFE:=0}"
: "${MULTI_SHARD:=0}"
: "${JOB_SCRIPT:=jobs/generate_per_question.pbs}"
: "${REPAIR_SUBDIR:=repair}"
: "${SHARD_ID_PREFIX:=repair}"
: "${EXCLUDE_SHARD_IDS:=}"
: "${EXCLUDE_QUESTION_IDS:=}"

: "${SCRATCH:?SCRATCH must be available from .env for per-question repair submission}"
RUN_DIR="${SCRATCH}/runs/${RUN_NAME}"
REPAIR_DIR="${RUN_DIR}/${REPAIR_SUBDIR}"
REPORT_PATH="${REPAIR_DIR}/repair_report.json"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "per-question run directory not found: ${RUN_DIR}"
  exit 1
fi

build_cmd=(
  python
  scripts/build_per_question_repair.py
  --run-dir "${RUN_DIR}"
  --output-dir "${REPAIR_DIR}"
  --target-traces-per-shard "${TARGET_TRACES_PER_SHARD}"
)

if [[ -n "${QUESTIONS_PER_SHARD}" ]]; then
  build_cmd+=(--questions-per-shard "${QUESTIONS_PER_SHARD}")
fi

if [[ "${ALLOW_APPEND_UNSAFE}" == "1" ]]; then
  build_cmd+=(--include-append-unsafe)
fi

if [[ "${MULTI_SHARD}" == "1" ]]; then
  build_cmd+=(--multi-shard)
fi

if [[ -n "${EXCLUDE_SHARD_IDS}" ]]; then
  IFS=',' read -r -a excluded_shard_ids <<< "${EXCLUDE_SHARD_IDS}"
  for shard_id in "${excluded_shard_ids[@]}"; do
    [[ -n "${shard_id}" ]] && build_cmd+=(--exclude-shard-id "${shard_id}")
  done
fi

if [[ -n "${EXCLUDE_QUESTION_IDS}" ]]; then
  IFS=',' read -r -a excluded_question_ids <<< "${EXCLUDE_QUESTION_IDS}"
  for question_id in "${excluded_question_ids[@]}"; do
    [[ -n "${question_id}" ]] && build_cmd+=(--exclude-question-id "${question_id}")
  done
fi

"${build_cmd[@]}"

if [[ ! -f "${REPORT_PATH}" ]]; then
  echo "missing repair report at ${REPORT_PATH}"
  exit 1
fi

repair_manifest_count=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['repair_manifest_count'])" "${REPORT_PATH}")
repair_manifest_path=$(python -c "import json,sys; value=json.load(open(sys.argv[1], encoding='utf-8')).get('repair_manifest_path') or ''; print(value)" "${REPORT_PATH}")
append_safe_issue_count=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['append_safe_issue_count'])" "${REPORT_PATH}")
append_unsafe_issue_count=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['append_unsafe_issue_count'])" "${REPORT_PATH}")
excluded_issue_count=$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['excluded_issue_count'])" "${REPORT_PATH}")

echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "repair_dir=${REPAIR_DIR}"
echo "report_path=${REPORT_PATH}"
echo "repair_manifest_count=${repair_manifest_count}"
echo "append_safe_issue_count=${append_safe_issue_count}"
echo "append_unsafe_issue_count=${append_unsafe_issue_count}"
echo "excluded_issue_count=${excluded_issue_count}"
echo "exclude_shard_ids=${EXCLUDE_SHARD_IDS:-<none>}"
echo "exclude_question_ids=${EXCLUDE_QUESTION_IDS:-<none>}"
echo "repair_manifest_path=${repair_manifest_path:-<none>}"
echo "job_script=${JOB_SCRIPT}"

if (( repair_manifest_count == 0 )); then
  echo "No append-safe repair questions were found. Nothing to submit."
  exit 0
fi

if [[ -z "${repair_manifest_path}" || ! -f "${repair_manifest_path}" ]]; then
  echo "repair manifest missing: ${repair_manifest_path}"
  exit 1
fi

if [[ "${MULTI_SHARD}" == "1" ]]; then
  shard_plan_cmd=(
    python
    scripts/build_per_question_shard_plan.py
    --question-manifest "${repair_manifest_path}"
    --target-traces-per-shard "${TARGET_TRACES_PER_SHARD}"
    --format tsv
  )
  if [[ -n "${QUESTIONS_PER_SHARD}" ]]; then
    shard_plan_cmd+=(--questions-per-shard "${QUESTIONS_PER_SHARD}")
  fi
  mapfile -t shard_rows < <("${shard_plan_cmd[@]}")

  if (( ${#shard_rows[@]} == 0 )); then
    echo "failed to build repair shard plan"
    exit 1
  fi

  num_shards=${#shard_rows[@]}
  for shard_row in "${shard_rows[@]}"; do
    IFS=$'\t' read -r shard_index start_idx end_idx question_count shard_traces <<< "${shard_row}"
    shard_id="${SHARD_ID_PREFIX}-$(printf '%04d_%04d' "${start_idx}" "${end_idx}")-$(date +%m%d_%H%M%S)"
    echo "submitting repair shard=$(( shard_index + 1 ))/${num_shards} start=${start_idx} end=${end_idx} questions=${question_count} target_traces=${shard_traces} shard_id=${shard_id}"
    qsub -v RUN_NAME="${RUN_NAME}",CONFIG_PATH="${CONFIG_PATH}",PROMPTS_DIR="${PROMPTS_DIR}",QUESTION_MANIFEST_PATH="${repair_manifest_path}",START_IDX="${start_idx}",END_IDX="${end_idx}",SHARD_ID="${shard_id}",PRESERVE_ROOT_SELECTION_INPUTS="1" "${JOB_SCRIPT}"
  done
  exit 0
fi

shard_id="${SHARD_ID_PREFIX}-$(date +%m%d_%H%M%S)"
echo "submitting repair shard start=0 end=${repair_manifest_count} shard_id=${shard_id}"
qsub -v RUN_NAME="${RUN_NAME}",CONFIG_PATH="${CONFIG_PATH}",PROMPTS_DIR="${PROMPTS_DIR}",QUESTION_MANIFEST_PATH="${repair_manifest_path}",START_IDX="0",END_IDX="${repair_manifest_count}",SHARD_ID="${shard_id}",PRESERVE_ROOT_SELECTION_INPUTS="1" "${JOB_SCRIPT}"
