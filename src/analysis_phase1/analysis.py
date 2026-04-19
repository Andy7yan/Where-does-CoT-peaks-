"""Analysis helpers for the canonical difficulty/bin/sample workflow."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from src.analysis_phase1.nldd_measurement import (
    build_correct_token_ids,
    compute_logit_margin,
    extract_trace_horizon,
    measure_nldd,
)
from src.analysis_phase1.nldd_shared import _flatten_numeric_values, _write_jsonl
from src.analysis_phase1.io import SampleRecord, load_analysis_samples, load_analysis_traces_by_difficulty
from src.data_phase1.prompting import build_nldd_clean_prompt
from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER


def build_prompt_hidden_state_fn(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any,
    layer: str,
) -> Callable[[str], Any]:
    """Create a prompt -> last-token hidden-state extractor."""

    def encode_prompt(prompt: str) -> Any:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prepared_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        with torch_module.no_grad():
            outputs = model(
                **prepared_inputs,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model backend did not return hidden states.")
        layer_index = _resolve_hidden_layer_index(layer, len(hidden_states))
        return hidden_states[layer_index][0, -1, :]

    return encode_prompt


def build_trace_trajectory_fn(
    *,
    prompt_hidden_state_fn: Callable[[str], Any],
) -> Callable[[str, Sequence[str]], list[Any]]:
    """Create a `(question, steps)` -> prefix-trajectory encoder."""

    def encode_trace(question: str, steps: Sequence[str]) -> list[Any]:
        trajectory = [
            prompt_hidden_state_fn(build_nldd_clean_prompt(question=question, steps=[]))
        ]
        trajectory.extend(
            prompt_hidden_state_fn(build_nldd_clean_prompt(question=question, steps=list(steps[:prefix_length])))
            for prefix_length in range(1, len(steps) + 1)
        )
        return trajectory

    return encode_trace


def calibrate_s_on_samples(
    samples: Sequence[SampleRecord],
    *,
    prompt_logits_fn: Callable[[str], Any],
) -> float:
    """Calibrate S on all selected clean samples in the canonical layout."""

    std_values: list[float] = []
    for sample in samples:
        if not sample.clean_steps:
            continue
        prompt = build_nldd_clean_prompt(
            question=sample.question_text,
            steps=list(sample.clean_steps),
        )
        logits = prompt_logits_fn(prompt)
        std_values.append(_compute_vector_std(logits))
    if not std_values:
        raise ValueError("Cannot calibrate S without at least one non-empty clean sample.")
    return sum(std_values) / len(std_values)


def measure_sample_nldd(
    sample: SampleRecord,
    *,
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    s_value: float,
    ld_epsilon: float,
) -> list[dict[str, Any]]:
    """Measure all NLDD rows for one complete sample."""

    if not sample.clean_steps:
        return []

    clean_prompt = build_nldd_clean_prompt(
        question=sample.question_text,
        steps=list(sample.clean_steps),
    )
    correct_token_ids = build_correct_token_ids(sample.gold_answer, tokenizer)
    clean_logits = prompt_logits_fn(clean_prompt)
    ld_clean = compute_logit_margin(clean_logits, correct_token_ids, s_value)
    low_ld_clean = abs(ld_clean) < ld_epsilon

    rows: list[dict[str, Any]] = []
    for k in sample.k_values:
        corrupt_payload = sample.corruptions_by_k[k]
        corrupt_prompt = build_nldd_clean_prompt(
            question=sample.question_text,
            steps=[str(step) for step in corrupt_payload.get("steps", [])],
        )
        ld_corrupt: float | None = None
        nldd_value: float | None = None
        exclusion_reason: str | None = None
        if low_ld_clean:
            exclusion_reason = "low_ld_clean"
        else:
            corrupt_logits = prompt_logits_fn(corrupt_prompt)
            ld_corrupt = compute_logit_margin(corrupt_logits, correct_token_ids, s_value)
            nldd_value = measure_nldd(ld_clean, ld_corrupt, ld_epsilon=ld_epsilon)
            if nldd_value is None:
                exclusion_reason = "low_ld_clean"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_trace_id": sample.source_trace_id,
                "question_id": sample.question_id,
                "difficulty": sample.difficulty,
                "length": sample.length,
                "k": k,
                "trace_tier": sample.trace_tier,
                "corruption_tier": int(sample.per_k_meta[k]["tier"]),
                "corruption_id": sample.per_k_meta[k]["corruption_id"],
                "ld_clean": ld_clean,
                "ld_corrupt": ld_corrupt,
                "nldd_value": nldd_value,
                "measurement_exclusion_reason": exclusion_reason,
            }
        )

    horizon = extract_trace_horizon(
        [
            {
                "corruption_step_index": row["k"],
                "actual_clean_length": sample.length,
                "nldd_value": row["nldd_value"],
            }
            for row in rows
        ]
    )
    for row in rows:
        row["k_star_trace"] = horizon["k_star_trace"]
        row["r_star_trace"] = horizon["r_star_trace"]
    return rows


def measure_sample_tas(
    sample: SampleRecord,
    *,
    trace_trajectory_fn: Callable[[str, Sequence[str]], list[Any]],
    plateau_threshold: float | None,
) -> dict[str, Any]:
    """Measure TAS on one clean sample from its prefix trajectory."""

    vectors = trace_trajectory_fn(sample.question_text, sample.clean_steps)
    tas_value, plateau_step = compute_tas_from_vectors(vectors, plateau_threshold=plateau_threshold)
    return {
        "sample_id": sample.sample_id,
        "source_trace_id": sample.source_trace_id,
        "question_id": sample.question_id,
        "difficulty": sample.difficulty,
        "length": sample.length,
        "trace_tier": sample.trace_tier,
        "tas_value": tas_value,
        "plateau_step": plateau_step,
        "num_prefixes": max(len(vectors) - 1, 0),
    }


def measure_sample_tas_curve(
    sample: SampleRecord,
    *,
    trace_trajectory_fn: Callable[[str, Sequence[str]], list[Any]],
    plateau_threshold: float | None,
) -> list[dict[str, Any]]:
    """Measure the prefix-by-prefix TAS curve for one clean sample."""

    vectors = trace_trajectory_fn(sample.question_text, sample.clean_steps)
    curve = compute_tas_curve_from_vectors(vectors, plateau_threshold=plateau_threshold)
    return [
        {
            "sample_id": sample.sample_id,
            "source_trace_id": sample.source_trace_id,
            "question_id": sample.question_id,
            "difficulty": sample.difficulty,
            "length": sample.length,
            "trace_tier": sample.trace_tier,
            "step_index": point["step_index"],
            "tas_value": point["tas_value"],
            "plateau_step": point["plateau_step"],
        }
        for point in curve
    ]


def compute_tas_from_vectors(
    vectors: Sequence[Any],
    *,
    plateau_threshold: float | None,
) -> tuple[float, int | None]:
    """Compute TAS as geometric efficiency over the prefix trajectory."""

    flattened = [_flatten_numeric_values(vector) for vector in vectors]
    if len(flattened) <= 1:
        return 0.0, None

    step_lengths = [
        _l2_distance(flattened[index], flattened[index + 1])
        for index in range(len(flattened) - 1)
    ]
    path_length = sum(step_lengths)
    if path_length <= 0.0:
        return 0.0, None

    displacement = _l2_distance(flattened[0], flattened[-1])
    tas_value = displacement / path_length
    plateau_step = None
    if plateau_threshold is not None:
        running_values: list[float] = []
        for index, step_length in enumerate(step_lengths, start=2):
            running_values.append(step_length)
            if len(running_values) >= 2 and abs(running_values[-1] - running_values[-2]) <= plateau_threshold:
                plateau_step = index
                break
    return tas_value, plateau_step


def compute_tas_curve_from_vectors(
    vectors: Sequence[Any],
    *,
    plateau_threshold: float | None,
) -> list[dict[str, Any]]:
    """Compute TAS for each prefix endpoint, treating that endpoint as the final state."""

    flattened = [_flatten_numeric_values(vector) for vector in vectors]
    if len(flattened) <= 1:
        return []

    curve: list[dict[str, Any]] = []
    for step_index in range(1, len(flattened)):
        tas_value, plateau_step = compute_tas_from_vectors(
            flattened[: step_index + 1],
            plateau_threshold=plateau_threshold,
        )
        curve.append(
            {
                "step_index": step_index,
                "tas_value": tas_value,
                "plateau_step": plateau_step,
            }
        )
    return curve


def run_analysis(
    *,
    run_dir: str,
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    trace_trajectory_fn: Callable[[str, Sequence[str]], list[Any]],
    ld_epsilon: float,
    tas_plateau_threshold: float | None,
) -> dict[str, Any]:
    """Run the analysis flow over the canonical difficulty/sample layout."""

    run_path = Path(run_dir)
    analysis_dir = run_path / "analysis_phase1"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    samples = load_analysis_samples(run_path)
    traces_by_difficulty = load_analysis_traces_by_difficulty(run_path)
    if not samples:
        raise ValueError("No analysis samples were discovered under difficulty/*/bins/*/samples.")

    s_value = calibrate_s_on_samples(samples, prompt_logits_fn=prompt_logits_fn)
    nldd_rows: list[dict[str, Any]] = []
    tas_rows: list[dict[str, Any]] = []
    tas_curve_rows: list[dict[str, Any]] = []
    for sample in samples:
        nldd_rows.extend(
            measure_sample_nldd(
                sample,
                prompt_logits_fn=prompt_logits_fn,
                tokenizer=tokenizer,
                s_value=s_value,
                ld_epsilon=ld_epsilon,
            )
        )
        tas_rows.append(
            measure_sample_tas(
                sample,
                trace_trajectory_fn=trace_trajectory_fn,
                plateau_threshold=tas_plateau_threshold,
            )
        )
        tas_curve_rows.extend(
            measure_sample_tas_curve(
                sample,
                trace_trajectory_fn=trace_trajectory_fn,
                plateau_threshold=tas_plateau_threshold,
            )
        )

    accuracy_rows = build_accuracy_by_length_rows(traces_by_difficulty)
    lstar_rows = build_lstar_rows(accuracy_rows)
    nldd_surface_rows = build_nldd_surface_rows(nldd_rows)
    kstar_rows = build_kstar_rows(nldd_surface_rows)
    tas_curve_summary_rows = build_tas_curve_summary_rows(tas_curve_rows)
    bin_status_rows = build_bin_status_rows(traces_by_difficulty, samples)
    failure_rows = build_failure_stats_rows(nldd_rows=nldd_rows, bin_status_rows=bin_status_rows)

    accuracy_path = analysis_dir / "accuracy_by_length.csv"
    s_calibration_path = analysis_dir / "S_calibration.json"
    nldd_per_trace_path = analysis_dir / "nldd_per_trace.jsonl"
    tas_per_trace_path = analysis_dir / "tas_per_trace.jsonl"
    tas_curve_per_trace_path = analysis_dir / "tas_curve_per_trace.jsonl"
    nldd_surface_path = analysis_dir / "nldd_surface.csv"
    tas_curve_path = analysis_dir / "tas_curve.csv"
    k_star_by_l_path = analysis_dir / "k_star_by_L.csv"
    l_star_path = analysis_dir / "L_star.csv"
    bin_status_path = analysis_dir / "bin_status.csv"
    failure_stats_path = analysis_dir / "failure_stats.csv"

    _write_csv(
        accuracy_path,
        rows=accuracy_rows,
        fieldnames=["difficulty", "length", "n", "mean_accuracy", "se_accuracy"],
    )
    _write_json(
        s_calibration_path,
        {
            "schema_version": "stage1_analysis_s_calibration_v1",
            "s_value": s_value,
            "trace_count": len(samples),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_jsonl(nldd_per_trace_path, nldd_rows)
    _write_jsonl(tas_per_trace_path, tas_rows)
    _write_jsonl(tas_curve_per_trace_path, tas_curve_rows)
    _write_csv(
        nldd_surface_path,
        rows=nldd_surface_rows,
        fieldnames=["difficulty", "length", "k", "n", "mean_nldd", "se_nldd"],
    )
    _write_csv(
        tas_curve_path,
        rows=tas_curve_summary_rows,
        fieldnames=["difficulty", "length", "step_index", "n", "mean_tas", "se_tas"],
    )
    _write_csv(
        k_star_by_l_path,
        rows=kstar_rows,
        fieldnames=["difficulty", "length", "k_star", "mean_nldd", "n"],
    )
    _write_csv(
        l_star_path,
        rows=lstar_rows,
        fieldnames=["difficulty", "L_star", "mean_accuracy", "n"],
    )
    _write_csv(
        bin_status_path,
        rows=bin_status_rows,
        fieldnames=[
            "difficulty",
            "length",
            "trace_total",
            "eligible_clean_traces",
            "selected_samples",
            "tier1_samples",
            "tier2_samples",
            "status",
        ],
    )
    _write_csv(
        failure_stats_path,
        rows=failure_rows,
        fieldnames=["category", "key", "count"],
    )

    _validate_analysis_outputs(
        [
            accuracy_path,
            s_calibration_path,
            nldd_per_trace_path,
            tas_per_trace_path,
            tas_curve_per_trace_path,
            nldd_surface_path,
            tas_curve_path,
            k_star_by_l_path,
            l_star_path,
            bin_status_path,
            failure_stats_path,
        ]
    )

    return {
        "analysis_dir": str(analysis_dir),
        "sample_count": len(samples),
        "s_value": s_value,
        "accuracy_by_length_path": str(accuracy_path),
        "s_calibration_path": str(s_calibration_path),
        "nldd_per_trace_path": str(nldd_per_trace_path),
        "tas_per_trace_path": str(tas_per_trace_path),
        "tas_curve_per_trace_path": str(tas_curve_per_trace_path),
        "nldd_surface_path": str(nldd_surface_path),
        "tas_curve_path": str(tas_curve_path),
        "k_star_by_L_path": str(k_star_by_l_path),
        "L_star_path": str(l_star_path),
        "bin_status_path": str(bin_status_path),
        "failure_stats_path": str(failure_stats_path),
    }


def build_accuracy_by_length_rows(
    traces_by_difficulty: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate per-difficulty accuracy by exact clean length."""

    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTY_ORDER:
        grouped: dict[int, list[int]] = {}
        for trace in traces_by_difficulty.get(difficulty, []):
            grouped.setdefault(int(trace["actual_num_steps"]), []).append(1 if trace["is_correct"] else 0)
        for length in sorted(grouped):
            outcomes = grouped[length]
            score = sum(outcomes) / len(outcomes)
            rows.append(
                {
                    "difficulty": difficulty,
                    "length": length,
                    "n": len(outcomes),
                    "mean_accuracy": score,
                    "se_accuracy": _standard_error(outcomes),
                }
            )
    return rows


def build_lstar_rows(accuracy_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve L* per difficulty from the accuracy-by-length table."""

    rows_by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for row in accuracy_rows:
        rows_by_difficulty.setdefault(str(row["difficulty"]), []).append(dict(row))

    lstar_rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTY_ORDER:
        difficulty_rows = rows_by_difficulty.get(difficulty, [])
        if not difficulty_rows:
            continue
        best_row = max(
            difficulty_rows,
            key=lambda row: (float(row["mean_accuracy"]), -int(row["length"])),
        )
        lstar_rows.append(
            {
                "difficulty": difficulty,
                "L_star": int(best_row["length"]),
                "mean_accuracy": float(best_row["mean_accuracy"]),
                "n": int(best_row["n"]),
            }
        )
    return lstar_rows


def build_nldd_surface_rows(nldd_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-(difficulty, L, k) NLDD means and standard errors."""

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in nldd_rows:
        value = row.get("nldd_value")
        if value is None:
            continue
        key = (str(row["difficulty"]), int(row["length"]), int(row["k"]))
        grouped.setdefault(key, []).append(float(value))

    surface_rows: list[dict[str, Any]] = []
    for difficulty, length, k in sorted(
        grouped,
        key=lambda item: (DIFFICULTY_ORDER.index(item[0]), item[1], item[2]),
    ):
        values = grouped[(difficulty, length, k)]
        surface_rows.append(
            {
                "difficulty": difficulty,
                "length": length,
                "k": k,
                "n": len(values),
                "mean_nldd": mean(values),
                "se_nldd": _standard_error(values),
            }
        )
    return surface_rows


def build_kstar_rows(surface_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve k*(L) from the aggregated NLDD surface."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in surface_rows:
        grouped.setdefault((str(row["difficulty"]), int(row["length"])), []).append(dict(row))

    kstar_rows: list[dict[str, Any]] = []
    for difficulty, length in sorted(
        grouped,
        key=lambda item: (DIFFICULTY_ORDER.index(item[0]), item[1]),
    ):
        rows = grouped[(difficulty, length)]
        best_row = max(
            rows,
            key=lambda row: (float(row["mean_nldd"]), -int(row["k"])),
        )
        kstar_rows.append(
            {
                "difficulty": difficulty,
                "length": length,
                "k_star": int(best_row["k"]),
                "mean_nldd": float(best_row["mean_nldd"]),
                "n": int(best_row["n"]),
            }
        )
    return kstar_rows


def build_tas_curve_summary_rows(tas_curve_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate TAS curves by difficulty, exact length, and prefix step index."""

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in tas_curve_rows:
        key = (str(row["difficulty"]), int(row["length"]), int(row["step_index"]))
        grouped.setdefault(key, []).append(float(row["tas_value"]))

    summary_rows: list[dict[str, Any]] = []
    for difficulty, length, step_index in sorted(
        grouped,
        key=lambda item: (DIFFICULTY_ORDER.index(item[0]), item[1], item[2]),
    ):
        values = grouped[(difficulty, length, step_index)]
        summary_rows.append(
            {
                "difficulty": difficulty,
                "length": length,
                "step_index": step_index,
                "n": len(values),
                "mean_tas": mean(values),
                "se_tas": _standard_error(values),
            }
        )
    return summary_rows


def build_bin_status_rows(
    traces_by_difficulty: dict[str, list[dict[str, Any]]],
    samples: Sequence[SampleRecord],
) -> list[dict[str, Any]]:
    """Summarize sample selection coverage per (difficulty, length) bin."""

    selected_by_bin: dict[tuple[str, int], list[SampleRecord]] = {}
    for sample in samples:
        selected_by_bin.setdefault((sample.difficulty, sample.length), []).append(sample)

    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTY_ORDER:
        grouped_traces: dict[int, list[dict[str, Any]]] = {}
        for trace in traces_by_difficulty.get(difficulty, []):
            grouped_traces.setdefault(int(trace["actual_num_steps"]), []).append(trace)
        all_lengths = sorted(set(grouped_traces) | {length for d, length in selected_by_bin if d == difficulty})
        for length in all_lengths:
            traces = grouped_traces.get(length, [])
            eligible_clean = [
                trace
                for trace in traces
                if trace.get("is_correct") and trace.get("nldd_measurement_eligible", True)
            ]
            selected = selected_by_bin.get((difficulty, length), [])
            tier1 = sum(1 for sample in selected if sample.trace_tier == 1)
            tier2 = sum(1 for sample in selected if sample.trace_tier == 2)
            rows.append(
                {
                    "difficulty": difficulty,
                    "length": length,
                    "trace_total": len(traces),
                    "eligible_clean_traces": len(eligible_clean),
                    "selected_samples": len(selected),
                    "tier1_samples": tier1,
                    "tier2_samples": tier2,
                    "status": "ok" if selected else "insufficient",
                }
            )
    return rows


def build_failure_stats_rows(
    *,
    nldd_rows: Sequence[dict[str, Any]],
    bin_status_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a compact failure summary for downstream reporting."""

    low_ld_count = sum(1 for row in nldd_rows if row.get("measurement_exclusion_reason") == "low_ld_clean")
    null_nldd_count = sum(1 for row in nldd_rows if row.get("nldd_value") is None)
    insufficient_bins = sum(1 for row in bin_status_rows if row["status"] != "ok")
    return [
        {"category": "nldd", "key": "low_ld_clean_rows", "count": low_ld_count},
        {"category": "nldd", "key": "null_nldd_rows", "count": null_nldd_count},
        {"category": "bins", "key": "insufficient_bins", "count": insufficient_bins},
    ]


def _validate_analysis_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected analysis output was not written: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Analysis output is empty: {path}")


def _resolve_hidden_layer_index(layer: str, num_hidden_states: int) -> int:
    if num_hidden_states <= 0:
        raise ValueError("num_hidden_states must be positive.")
    normalized = layer.strip().lower()
    if normalized == "middle":
        return max((num_hidden_states - 1) // 2, 0)
    try:
        index = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported TAS layer selector: {layer!r}") from exc
    if index < 0 or index >= num_hidden_states:
        raise IndexError(f"Hidden-state layer index {index} is out of bounds for {num_hidden_states} states.")
    return index


def _compute_vector_std(logits: Any) -> float:
    values = _flatten_numeric_values(logits)
    if not values:
        raise ValueError("Cannot compute standard deviation for an empty vector.")
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("L2 distance expects vectors of equal length.")
    return math.sqrt(sum((l - r) ** 2 for l, r in zip(left, right)))


def _standard_error(values: Sequence[float | int]) -> float:
    if len(values) <= 1:
        return 0.0
    numeric = [float(value) for value in values]
    avg = sum(numeric) / len(numeric)
    variance = sum((value - avg) ** 2 for value in numeric) / (len(numeric) - 1)
    return math.sqrt(variance / len(numeric))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, *, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
