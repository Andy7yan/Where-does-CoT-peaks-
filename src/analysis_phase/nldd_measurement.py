"""Logit-scoring and measurement helpers for the NLDD workflow."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

from src.common.corruption import (
    DEFAULT_FLOAT_PERTURBATION_RANGE,
    DEFAULT_INTEGER_PERTURBATION_RANGE,
    corrupt_step_text_with_fallbacks,
)
from src.data_phase1.prompting import build_nldd_clean_prompt
from src.data_phase2.pipeline import load_stage1_traces

from src.analysis_phase.nldd_corruption import build_corruption_record, summarize_corruption_records
from src.analysis_phase.nldd_shared import (
    _compute_vector_std,
    _flatten_numeric_values,
    _flatten_token_ids,
    _format_gold_answer_variants,
    _move_model_inputs_to_device,
    _stable_seed,
    _write_jsonl,
)


def build_prompt_logit_fn(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any,
) -> Callable[[str], Any]:
    """Create a prompt -> final-token-logits scorer."""

    def score_prompt(prompt: str) -> Any:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prepared_inputs = _move_model_inputs_to_device(model_inputs, device)
        with torch_module.no_grad():
            outputs = model(**prepared_inputs)
        return outputs.logits[0, -1, :]

    return score_prompt


def calibrate_s(
    clean_traces: list[dict[str, Any]],
    *,
    prompt_logits_fn: Callable[[str], Any],
) -> float:
    """Calibrate the global normalization constant S on all correct traces."""

    std_values: list[float] = []
    for trace in clean_traces:
        if not trace.get("is_correct"):
            continue
        steps = [str(step) for step in trace.get("steps", [])]
        if not steps:
            continue
        prompt = build_nldd_clean_prompt(
            question=str(trace["question_text"]),
            steps=steps,
        )
        logits = prompt_logits_fn(prompt)
        std_values.append(_compute_vector_std(logits))

    if not std_values:
        raise ValueError("Cannot calibrate S without at least one correct trace with steps.")
    return sum(std_values) / len(std_values)


def compute_logit_margin(
    logits: object,
    correct_token_ids: Sequence[int],
    s_value: float,
) -> float:
    """Compute the standardized LD margin for one final-token logit vector."""

    if s_value <= 0:
        raise ValueError("s_value must be positive.")
    if not correct_token_ids:
        raise ValueError("correct_token_ids must not be empty.")

    values = _flatten_numeric_values(logits)
    vocab_size = len(values)
    normalized_ids = sorted({int(token_id) for token_id in correct_token_ids})
    invalid_ids = [token_id for token_id in normalized_ids if token_id < 0 or token_id >= vocab_size]
    if invalid_ids:
        raise IndexError(f"Correct token ids out of bounds for logits size {vocab_size}: {invalid_ids[:5]}")
    if len(normalized_ids) >= vocab_size:
        raise ValueError("correct_token_ids must not cover the entire vocabulary.")

    correct_max = max(values[token_id] for token_id in normalized_ids)
    correct_id_set = set(normalized_ids)
    incorrect_max = max(
        value
        for index, value in enumerate(values)
        if index not in correct_id_set
    )
    return (correct_max - incorrect_max) / s_value


def measure_nldd(
    clean_margin: float,
    corrupt_margin: float,
    *,
    ld_epsilon: float,
) -> float | None:
    """Compute the NLDD percentage drop when the clean LD is stable enough."""

    if abs(clean_margin) < ld_epsilon:
        return None
    return ((clean_margin - corrupt_margin) / abs(clean_margin)) * 100.0


def build_correct_token_ids(gold_answer: float | int | str, tokenizer: Any) -> list[int]:
    """Build the candidate correct-answer token ids for first-token LD scoring."""

    variants = _format_gold_answer_variants(gold_answer)
    token_ids: set[int] = set()
    for variant in variants:
        encoded = tokenizer(
            variant,
            add_special_tokens=False,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
        flattened = _flatten_token_ids(input_ids)
        if flattened:
            token_ids.add(int(flattened[0]))
    if not token_ids:
        raise ValueError(f"Could not tokenize any correct-answer variants for gold answer {gold_answer!r}.")
    return sorted(token_ids)


def measure_trace_profile(
    *,
    trace: dict[str, Any],
    selection_row: dict[str, Any],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    s_value: float,
    ld_epsilon: float,
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> list[dict[str, Any]]:
    """Measure the full NLDD profile for one selected clean trace."""

    steps = [str(step) for step in trace.get("steps", [])]
    actual_clean_length = int(trace.get("actual_num_steps", len(steps)))
    if not steps or actual_clean_length <= 0:
        return []

    correct_token_ids = build_correct_token_ids(trace["gold_answer"], tokenizer)
    clean_prompt = build_nldd_clean_prompt(
        question=str(trace["question_text"]),
        steps=steps,
    )
    clean_logits = prompt_logits_fn(clean_prompt)
    ld_clean = compute_logit_margin(clean_logits, correct_token_ids, s_value)
    low_ld_clean = abs(ld_clean) < ld_epsilon

    rows: list[dict[str, Any]] = []
    for step_index in range(1, actual_clean_length + 1):
        step_text = steps[step_index - 1]
        rng = random.Random(_stable_seed(f"{seed}:full:{trace['trace_id']}:{step_index}"))
        corruption = corrupt_step_text_with_fallbacks(
            step_text,
            rng=rng,
            integer_perturbation_range=integer_perturbation_range,
            float_perturbation_range=float_perturbation_range,
            enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            token_counter=token_counter,
            token_delta_max=token_delta_max,
            retry_limit=retry_limit,
            max_perplexity_ratio=None,
        )
        corruption_record = build_corruption_record(
            shard_id="root",
            trace=trace,
            step_index=step_index,
            clean_step=step_text,
            corruption=corruption,
            selection_mode="full",
        )

        ld_corrupt: float | None = None
        nldd_value: float | None = None
        exclusion_reason: str | None = None
        if low_ld_clean:
            exclusion_reason = "low_ld_clean"
        elif corruption.corruption_failed:
            exclusion_reason = "corruption_failed"
        else:
            corrupt_logits = prompt_logits_fn(str(corruption_record["corrupt_prompt"]))
            ld_corrupt = compute_logit_margin(corrupt_logits, correct_token_ids, s_value)
            nldd_value = measure_nldd(
                ld_clean,
                ld_corrupt,
                ld_epsilon=ld_epsilon,
            )

        rows.append(
            {
                "nldd_id": f"{trace['trace_id']}_step{step_index}",
                "question_id": str(trace["question_id"]),
                "clean_trace_id": str(trace["trace_id"]),
                "difficulty": str(selection_row["difficulty"]),
                "length_bin": str(selection_row["length_bin"]),
                "raw_length_bin": selection_row.get("raw_length_bin"),
                "selected_for_near_lstar": bool(selection_row["selected_for_near_lstar"]),
                "prompt_id": str(trace.get("prompt_id", "")),
                "actual_clean_length": actual_clean_length,
                "corruption_step_index": step_index,
                "corruption_tier": corruption.corruption_tier,
                "corruption_type": corruption.corruption_type,
                "corruption_failed": corruption.corruption_failed,
                "failure_tier": corruption.failure_tier,
                "ld_clean": ld_clean,
                "ld_corrupt": ld_corrupt,
                "nldd_value": nldd_value,
                "measurement_exclusion_reason": exclusion_reason,
                "selection_mode": selection_row.get("selection_mode"),
                "near_lstar_selection_mode": selection_row.get("near_lstar_selection_mode"),
                "corruption_id": corruption_record["corruption_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    horizon = extract_trace_horizon(rows)
    for row in rows:
        row["k_star_trace"] = horizon["k_star_trace"]
        row["r_star_trace"] = horizon["r_star_trace"]
    return rows


def extract_trace_horizon(profile_rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    """Locate the per-trace peak-based horizon using the v4 tie-breaking rule."""

    valid_rows = [
        row
        for row in profile_rows
        if row.get("nldd_value") is not None and int(row["corruption_step_index"]) > 1
    ]
    if not valid_rows:
        return {
            "k_star_trace": None,
            "r_star_trace": None,
        }

    best_row = max(
        valid_rows,
        key=lambda row: (float(row["nldd_value"]), -int(row["corruption_step_index"])),
    )
    k_star = int(best_row["corruption_step_index"])
    length = int(best_row["actual_clean_length"])
    return {
        "k_star_trace": k_star,
        "r_star_trace": (k_star / length) if length > 0 else None,
    }


def measure_selected_traces(
    *,
    traces_by_id: dict[str, dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    s_value: float,
    ld_epsilon: float,
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure all selected traces and return rows plus a compact summary."""

    measurement_rows: list[dict[str, Any]] = []
    selected_rows = [
        row
        for row in selection_rows
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    ]
    for selection_row in selected_rows:
        trace_id = str(selection_row["trace_id"])
        trace = traces_by_id.get(trace_id)
        if trace is None:
            raise KeyError(f"Trace selection references missing trace_id: {trace_id}")
        measurement_rows.extend(
            measure_trace_profile(
                trace=trace,
                selection_row=selection_row,
                prompt_logits_fn=prompt_logits_fn,
                tokenizer=tokenizer,
                token_counter=token_counter,
                s_value=s_value,
                ld_epsilon=ld_epsilon,
                seed=seed,
                token_delta_max=token_delta_max,
                retry_limit=retry_limit,
                integer_perturbation_range=integer_perturbation_range,
                float_perturbation_range=float_perturbation_range,
                enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            )
        )

    corruption_summary = summarize_corruption_records({"full": measurement_rows})["full"]
    trace_ids_with_low_ld = {
        str(row["clean_trace_id"])
        for row in measurement_rows
        if row.get("measurement_exclusion_reason") == "low_ld_clean"
    }
    summary = {
        "selected_trace_count": len(selected_rows),
        "selected_for_nldd_count": sum(1 for row in selected_rows if row["selected_for_nldd"]),
        "selected_for_near_lstar_count": sum(
            1 for row in selected_rows if row["selected_for_near_lstar"]
        ),
        "measured_row_count": len(measurement_rows),
        "valid_nldd_count": sum(1 for row in measurement_rows if row["nldd_value"] is not None),
        "low_ld_clean_trace_count": len(trace_ids_with_low_ld),
        "corruption": corruption_summary,
    }
    return measurement_rows, summary


def validate_nldd_full_records(
    *,
    selection_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Validate the Stage D v4 handoff invariants for nldd_full.jsonl."""

    if not selection_rows:
        raise ValueError("trace_selection.csv is empty.")
    if not records:
        raise ValueError("nldd_full.jsonl is empty.")

    expected_by_trace_id = {
        str(row["trace_id"]): int(row["actual_clean_length"])
        for row in selection_rows
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    }
    actual_steps_by_trace_id: dict[str, set[int]] = defaultdict(set)
    for row in records:
        trace_id = str(row["clean_trace_id"])
        actual_steps_by_trace_id[trace_id].add(int(row["corruption_step_index"]))

    missing_trace_ids = sorted(set(expected_by_trace_id) - set(actual_steps_by_trace_id))
    if missing_trace_ids:
        sample = ", ".join(missing_trace_ids[:5])
        raise ValueError(f"nldd_full.jsonl is missing selected traces: {sample}")

    for trace_id, expected_length in expected_by_trace_id.items():
        expected_steps = set(range(1, expected_length + 1))
        actual_steps = actual_steps_by_trace_id[trace_id]
        if actual_steps != expected_steps:
            missing_steps = sorted(expected_steps - actual_steps)
            extra_steps = sorted(actual_steps - expected_steps)
            raise ValueError(
                "nldd_full.jsonl does not cover the full sweep for trace "
                f"{trace_id}: missing={missing_steps[:5]}, extra={extra_steps[:5]}"
            )


def write_s_calibration(
    path: str | Path,
    *,
    s_value: float,
    trace_count: int,
) -> None:
    """Persist the calibrated S constant for downstream provenance."""

    calibration_path = Path(path)
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": "stage1_s_calibration_v1",
                "s_value": s_value,
                "trace_count": trace_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_nldd_full_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the Stage D v4 nldd_full.jsonl file."""

    _write_jsonl(Path(path), rows)


def compute_v4_measurement_artifacts(
    *,
    run_dir: str,
    question_metadata: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    ld_epsilon: float,
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> dict[str, Any]:
    """Compute and persist the v4 Stage D measurement artifacts for one run."""

    run_path = Path(run_dir)
    traces = load_stage1_traces(run_path)
    traces_by_id = {
        str(trace["trace_id"]): trace
        for trace in traces
    }
    correct_traces = [
        trace
        for trace in traces
        if trace.get("is_correct") and trace.get("steps")
    ]
    if not correct_traces:
        raise ValueError("Stage D2 requires at least one correct clean trace.")

    s_value = calibrate_s(
        correct_traces,
        prompt_logits_fn=prompt_logits_fn,
    )
    measurement_rows, measurement_summary = measure_selected_traces(
        traces_by_id=traces_by_id,
        selection_rows=selection_rows,
        prompt_logits_fn=prompt_logits_fn,
        tokenizer=tokenizer,
        token_counter=token_counter,
        s_value=s_value,
        ld_epsilon=ld_epsilon,
        seed=seed,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        integer_perturbation_range=integer_perturbation_range,
        float_perturbation_range=float_perturbation_range,
        enable_tier3_semantic_flip=enable_tier3_semantic_flip,
    )
    validate_nldd_full_records(
        selection_rows=selection_rows,
        records=measurement_rows,
    )

    s_path = run_path / "s_calibration.json"
    nldd_path = run_path / "nldd_full.jsonl"
    summary_path = run_path / "corruption_summary.json"
    write_s_calibration(
        s_path,
        s_value=s_value,
        trace_count=len(correct_traces),
    )
    write_nldd_full_records(nldd_path, measurement_rows)
    summary_payload = {
        "metadata": {
            "schema_version": "stage1_nldd_measurement_summary_v1",
            "run_dir": str(run_path),
            "selection_trace_count": len(selection_rows),
            "question_metadata_count": len(question_metadata),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": measurement_summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "s_value": s_value,
        "s_calibration_path": str(s_path),
        "nldd_full_path": str(nldd_path),
        "corruption_summary_path": str(summary_path),
        "measurement_summary": measurement_summary,
    }
