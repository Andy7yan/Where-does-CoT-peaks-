"""Per-question analysis outputs for dense Stage 1 runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from src.analysis_phase1.analysis import (
    calibrate_s_on_samples,
    measure_sample_nldd,
    measure_sample_tas_curve,
)
from src.analysis_phase1.io import SampleRecord
from src.analysis_phase1.pq_io import (
    PQ_ANALYSIS_DIRNAME,
    load_per_question_bin_summaries,
    load_per_question_lstar_payloads,
    load_per_question_metadata,
    load_per_question_samples,
)


PQ_PIPELINE_NAME = "pq"


def run_per_question_analysis(
    *,
    run_dir: str,
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    trace_trajectory_fn: Callable[[str, Sequence[str]], list[Any]],
    ld_epsilon: float,
    tas_plateau_threshold: float | None,
    min_kstar_bins: int,
) -> dict[str, Any]:
    """Run per-question analysis and write the pq_analysis output contract."""

    run_path = Path(run_dir)
    output_dir = run_path / PQ_ANALYSIS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_per_question_samples(run_path)
    if not samples:
        raise ValueError(
            "Per-question analysis requires at least one retained sample under per_question/*/bins/*/samples."
        )
    question_metadata_rows = load_per_question_metadata(run_path)
    lstar_payloads = load_per_question_lstar_payloads(run_path)
    bin_summaries = load_per_question_bin_summaries(run_path)

    question_metadata_by_id = {
        str(row["question_id"]): row
        for row in question_metadata_rows
    }
    s_value = calibrate_s_on_samples(
        samples,
        prompt_logits_fn=prompt_logits_fn,
    )

    nldd_rows: list[dict[str, Any]] = []
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
        tas_curve_rows.extend(
            measure_sample_tas_curve(
                sample,
                trace_trajectory_fn=trace_trajectory_fn,
                plateau_threshold=tas_plateau_threshold,
            )
        )

    step_surface_rows = build_step_surface_rows(
        nldd_rows=nldd_rows,
        tas_curve_rows=tas_curve_rows,
        bin_summaries=bin_summaries,
    )
    kstar_by_bin = build_kstar_by_bin(nldd_rows)
    kstar_ratio_rows = build_kstar_ratio_rows(
        kstar_by_bin=kstar_by_bin,
        bin_summaries=bin_summaries,
        question_metadata_by_id=question_metadata_by_id,
    )
    lstar_rows = build_lstar_difficulty_rows(
        lstar_payloads=lstar_payloads,
        question_metadata_by_id=question_metadata_by_id,
    )
    failure_rows = build_failure_rows(
        question_metadata_rows=question_metadata_rows,
        lstar_payloads=lstar_payloads,
        bin_summaries=bin_summaries,
        kstar_by_bin=kstar_by_bin,
        min_kstar_bins=min_kstar_bins,
    )

    _write_csv(
        output_dir / "t1b_step_surface.csv",
        rows=step_surface_rows,
        fieldnames=[
            "scope",
            "pipeline",
            "L",
            "step",
            "mean_nldd",
            "nldd_se",
            "mean_tas_t",
            "tas_t_se",
            "n_clean",
            "bin_status",
        ],
    )
    _write_csv(
        output_dir / "t1c_kstar_ratio.csv",
        rows=kstar_ratio_rows,
        fieldnames=[
            "question_id",
            "difficulty_score",
            "L",
            "k_star",
            "k_star_ratio",
            "n_clean",
        ],
    )
    _write_csv(
        output_dir / "t2b_lstar_difficulty.csv",
        rows=lstar_rows,
        fieldnames=[
            "question_id",
            "difficulty_score",
            "l_star_A",
            "l_star_S",
            "l_star_consistent",
        ],
    )
    _write_csv(
        output_dir / "bin_status.csv",
        rows=bin_summaries,
        fieldnames=[
            "scope",
            "pipeline",
            "L",
            "n_total_traces",
            "n_correct",
            "n_tier1",
            "n_tier2",
            "n_failed",
            "n_retained",
            "bin_status",
        ],
    )
    _write_csv(
        output_dir / "failure_stats.csv",
        rows=failure_rows,
        fieldnames=[
            "question_id",
            "acc_pq",
            "degenerate",
            "l_curve_insufficient",
            "k_star_insufficient",
            "n_valid_bins",
            "n_total_bins",
        ],
    )
    _write_json(
        output_dir / "S_calibration.json",
        {
            "pipeline": PQ_PIPELINE_NAME,
            "s_value": s_value,
            "trace_count": len(samples),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "pipeline": PQ_PIPELINE_NAME,
        "analysis_dir": str(output_dir),
        "sample_count": len(samples),
        "s_value": s_value,
        "t1b_step_surface_path": str(output_dir / "t1b_step_surface.csv"),
        "t1c_kstar_ratio_path": str(output_dir / "t1c_kstar_ratio.csv"),
        "t2b_lstar_difficulty_path": str(output_dir / "t2b_lstar_difficulty.csv"),
        "bin_status_path": str(output_dir / "bin_status.csv"),
        "failure_stats_path": str(output_dir / "failure_stats.csv"),
    }


def build_step_surface_rows(
    *,
    nldd_rows: Sequence[dict[str, Any]],
    tas_curve_rows: Sequence[dict[str, Any]],
    bin_summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate NLDD and TAS_t values into the unified per-question step surface."""

    nldd_by_step: dict[tuple[str, int, int], list[float]] = {}
    for row in nldd_rows:
        value = row.get("nldd_value")
        if value is None:
            continue
        key = (
            str(row["question_id"]),
            int(row["length"]),
            int(row["k"]),
        )
        nldd_by_step.setdefault(key, []).append(float(value))

    tas_by_step: dict[tuple[str, int, int], list[float]] = {}
    for row in tas_curve_rows:
        key = (
            str(row["question_id"]),
            int(row["length"]),
            int(row["step_index"]),
        )
        tas_by_step.setdefault(key, []).append(float(row["tas_value"]))

    surface_rows: list[dict[str, Any]] = []
    for summary in sorted(
        bin_summaries,
        key=lambda row: (str(row["scope"]), int(row["L"])),
    ):
        question_id = str(summary["scope"])
        length = int(summary["L"])
        for step in range(1, length + 1):
            nldd_values = nldd_by_step.get((question_id, length, step), [])
            tas_values = tas_by_step.get((question_id, length, step), [])
            surface_rows.append(
                {
                    "scope": question_id,
                    "pipeline": PQ_PIPELINE_NAME,
                    "L": length,
                    "step": step,
                    "mean_nldd": mean(nldd_values) if nldd_values else None,
                    "nldd_se": _standard_error(nldd_values) if nldd_values else None,
                    "mean_tas_t": mean(tas_values) if tas_values else None,
                    "tas_t_se": _standard_error(tas_values) if tas_values else None,
                    "n_clean": int(summary["n_retained"]),
                    "bin_status": str(summary["bin_status"]),
                }
            )
    return surface_rows


def build_kstar_by_bin(
    nldd_rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Resolve k* for each valid per-question length bin."""

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in nldd_rows:
        value = row.get("nldd_value")
        if value is None:
            continue
        key = (
            str(row["question_id"]),
            int(row["length"]),
            int(row["k"]),
        )
        grouped.setdefault(key, []).append(float(value))

    rows_by_bin: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (question_id, length, step), values in grouped.items():
        rows_by_bin.setdefault((question_id, length), []).append(
            {
                "k": step,
                "mean_nldd": mean(values),
                "n_clean": len(values),
            }
        )

    results: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in rows_by_bin.items():
        best_row = max(
            rows,
            key=lambda row: (float(row["mean_nldd"]), -int(row["k"])),
        )
        results[key] = {
            "k_star": int(best_row["k"]),
            "mean_nldd": float(best_row["mean_nldd"]),
            "n_clean": int(best_row["n_clean"]),
        }
    return results


def build_kstar_ratio_rows(
    *,
    kstar_by_bin: dict[tuple[str, int], dict[str, Any]],
    bin_summaries: Sequence[dict[str, Any]],
    question_metadata_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the per-question k*/L table for valid bins only."""

    rows: list[dict[str, Any]] = []
    for summary in sorted(
        bin_summaries,
        key=lambda row: (str(row["scope"]), int(row["L"])),
    ):
        if str(summary["bin_status"]) != "ok":
            continue
        question_id = str(summary["scope"])
        length = int(summary["L"])
        kstar = kstar_by_bin.get((question_id, length))
        if kstar is None:
            continue
        meta = question_metadata_by_id[question_id]
        rows.append(
            {
                "question_id": question_id,
                "difficulty_score": float(meta["difficulty_score"]),
                "L": length,
                "k_star": int(kstar["k_star"]),
                "k_star_ratio": float(kstar["k_star"]) / length,
                "n_clean": int(summary["n_retained"]),
            }
        )
    return rows


def build_lstar_difficulty_rows(
    *,
    lstar_payloads: dict[str, dict[str, Any]],
    question_metadata_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the per-question L* vs difficulty scatter input."""

    rows: list[dict[str, Any]] = []
    for question_id in sorted(lstar_payloads):
        payload = lstar_payloads[question_id]
        if payload.get("l_star_A") is None or payload.get("l_star_S") is None:
            continue
        meta = question_metadata_by_id[question_id]
        rows.append(
            {
                "question_id": question_id,
                "difficulty_score": float(meta["difficulty_score"]),
                "l_star_A": int(payload["l_star_A"]),
                "l_star_S": int(payload["l_star_S"]),
                "l_star_consistent": bool(payload["l_star_consistent"]),
            }
        )
    return rows


def build_failure_rows(
    *,
    question_metadata_rows: Sequence[dict[str, Any]],
    lstar_payloads: dict[str, dict[str, Any]],
    bin_summaries: Sequence[dict[str, Any]],
    kstar_by_bin: dict[tuple[str, int], dict[str, Any]],
    min_kstar_bins: int,
) -> list[dict[str, Any]]:
    """Build one diagnostic row per question."""

    summaries_by_question: dict[str, list[dict[str, Any]]] = {}
    for summary in bin_summaries:
        summaries_by_question.setdefault(str(summary["scope"]), []).append(dict(summary))

    rows: list[dict[str, Any]] = []
    for meta in sorted(question_metadata_rows, key=lambda row: str(row["question_id"])):
        question_id = str(meta["question_id"])
        question_summaries = summaries_by_question.get(question_id, [])
        n_valid_bins = sum(
            1 for row in question_summaries
            if str(row["bin_status"]) == "ok"
        )
        resolved_kstar_bins = sum(
            1 for row in question_summaries
            if (question_id, int(row["L"])) in kstar_by_bin
        )
        lstar_payload = lstar_payloads.get(
            question_id,
            {"l_curve_insufficient": True},
        )
        rows.append(
            {
                "question_id": question_id,
                "acc_pq": float(meta["acc_pq"]),
                "degenerate": bool(meta["degenerate"]),
                "l_curve_insufficient": bool(lstar_payload.get("l_curve_insufficient", True)),
                "k_star_insufficient": resolved_kstar_bins < min_kstar_bins,
                "n_valid_bins": n_valid_bins,
                "n_total_bins": len(question_summaries),
            }
        )
    return rows


def _standard_error(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, *, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "PQ_PIPELINE_NAME",
    "build_failure_rows",
    "build_kstar_by_bin",
    "build_kstar_ratio_rows",
    "build_lstar_difficulty_rows",
    "build_step_surface_rows",
    "run_per_question_analysis",
]
