"""Per-question data-phase pipeline for dense Stage 1 runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.analysis_phase1.nldd import CorruptionSelectionConfig, build_corruption_records
from src.common.settings import ExperimentConfig
from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    load_per_question_manifest,
)
from src.data_phase1.pilot import build_token_counter
from src.data_phase2.aggregation_core import standard_error
from src.data_phase2.difficulty_groups import (
    _stable_seed,
    _write_json,
    _write_jsonl,
    build_trace_sample_bundle,
)
from src.data_phase2.pipeline import load_stage1_traces


QUESTION_METADATA_FILENAME = "question_metadata.jsonl"
PER_QUESTION_DIRNAME = "per_question"
PQ_PIPELINE_NAME = "pq"


def aggregate_per_question_outputs(
    run_dir: str | Path,
    *,
    config_path: str = "configs/stage1_per_question.yaml",
) -> dict[str, Any]:
    """Build the per-question data-phase handoff for a dense PQ run."""

    config = ExperimentConfig.from_yaml(config_path)
    run_path = Path(run_dir)
    traces = load_stage1_traces(run_path)
    if not traces:
        raise ValueError(f"Per-question data phase requires traces under '{run_path}'.")

    manifest_path = run_path / PER_QUESTION_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing per-question manifest required for data-phase export: {manifest_path}"
        )
    manifest_rows = load_per_question_manifest(manifest_path)
    manifest_by_question = {
        str(row["question_id"]): dict(row)
        for row in manifest_rows
    }

    min_length = _coalesce_int(config.analysis.min_nldd_length, default=3)
    lcurve_min_bin_size = _coalesce_int(
        config.analysis.per_question_lcurve_min_bin_size,
        default=5,
    )
    min_retained_traces = _coalesce_int(
        config.analysis.per_question_min_retained_traces,
        default=5,
    )
    max_retained_traces = _coalesce_int(
        config.analysis.per_question_max_retained_traces,
        default=20,
    )
    smoothing_window = _coalesce_int(
        config.analysis.per_question_lstar_smoothing_window,
        default=3,
    )
    min_lcurve_bins = _coalesce_int(
        config.analysis.per_question_min_lcurve_bins,
        default=3,
    )
    _validate_positive_odd_window(smoothing_window)

    traces_by_question = _group_traces_by_question(traces)
    metadata_rows, missing_question_ids = build_per_question_metadata_rows(
        traces_by_question=traces_by_question,
        manifest_by_question=manifest_by_question,
    )

    metadata_path = run_path / QUESTION_METADATA_FILENAME
    _write_jsonl(metadata_path, metadata_rows)

    token_counter = build_token_counter(tokenizer=None, approximate=True)
    per_question_root = run_path / PER_QUESTION_DIRNAME
    per_question_root.mkdir(parents=True, exist_ok=True)

    question_summaries: dict[str, Any] = {}
    metadata_by_question = {
        str(row["question_id"]): row
        for row in metadata_rows
    }
    for question_id in sorted(metadata_by_question):
        question_dir = per_question_root / question_id
        question_dir.mkdir(parents=True, exist_ok=True)

        question_traces = traces_by_question.get(question_id, [])
        question_meta = metadata_by_question[question_id]
        difficulty = str(question_meta["difficulty"])

        lcurve_rows = build_per_question_lcurve_rows(
            question_id=question_id,
            traces=question_traces,
            min_length=min_length,
            min_bin_size=lcurve_min_bin_size,
        )
        _write_lcurve_csv(question_dir / "l_curve.csv", lcurve_rows)

        if bool(question_meta["degenerate"]):
            lstar_payload = build_lstar_payload(
                question_meta=question_meta,
                lcurve_rows=lcurve_rows,
                smoothing_window=smoothing_window,
                min_lcurve_bins=min_lcurve_bins,
            )
            _write_json(question_dir / "l_star.json", lstar_payload)
            question_summaries[question_id] = {
                "difficulty": difficulty,
                "degenerate": True,
                "valid_lcurve_bins": 0,
                "exported_bin_count": 0,
                "retained_sample_count": 0,
            }
            continue

        bin_summaries = export_per_question_bins(
            question_dir=question_dir,
            question_id=question_id,
            difficulty=difficulty,
            traces=question_traces,
            token_counter=token_counter,
            integer_perturbation_range=tuple(config.nldd.integer_perturbation_range),
            float_perturbation_range=tuple(config.nldd.float_perturbation_range),
            enable_tier3_semantic_flip=config.nldd.enable_tier3_semantic_flip,
            corruption_token_delta_max=config.nldd.corruption_token_delta_max,
            corruption_retry_limit=config.nldd.corruption_retry_limit,
            corruption_seed=config.experiment.seed,
            min_length=min_length,
            min_retained_traces=min_retained_traces,
            max_retained_traces=max_retained_traces,
        )
        lstar_payload = build_lstar_payload(
            question_meta=question_meta,
            lcurve_rows=lcurve_rows,
            smoothing_window=smoothing_window,
            min_lcurve_bins=min_lcurve_bins,
        )
        _write_json(question_dir / "l_star.json", lstar_payload)
        question_summaries[question_id] = {
            "difficulty": difficulty,
            "degenerate": False,
            "valid_lcurve_bins": len(lcurve_rows),
            "exported_bin_count": len(bin_summaries),
            "retained_sample_count": sum(
                int(row["n_retained"])
                for row in bin_summaries
            ),
        }

    return {
        "pipeline": PQ_PIPELINE_NAME,
        "run_dir": str(run_path),
        "traces_path": str(run_path / "traces.jsonl"),
        "question_metadata_path": str(metadata_path),
        "per_question_root": str(per_question_root),
        "question_count": len(metadata_rows),
        "trace_count": len(traces),
        "missing_question_count": len(missing_question_ids),
        "missing_question_ids": missing_question_ids,
        "questions": question_summaries,
    }


def build_per_question_metadata_rows(
    *,
    traces_by_question: dict[str, list[dict[str, Any]]],
    manifest_by_question: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build root metadata rows for each selected per-question item."""

    rows: list[dict[str, Any]] = []
    missing_question_ids: list[str] = []
    for question_id in sorted(manifest_by_question):
        question_traces = traces_by_question.get(question_id, [])
        if not question_traces:
            missing_question_ids.append(question_id)
            continue
        correct_count = sum(1 for trace in question_traces if bool(trace["is_correct"]))
        total_traces = len(question_traces)
        acc_pq = correct_count / total_traces if total_traces else 0.0
        length_distribution = Counter(
            int(trace["actual_num_steps"])
            for trace in question_traces
        )
        difficulty = str(manifest_by_question[question_id]["source_difficulty_bucket"])
        rows.append(
            {
                "question_id": question_id,
                "difficulty": difficulty,
                "acc_pq": acc_pq,
                "difficulty_score": 1.0 - acc_pq,
                "degenerate": acc_pq in {0.0, 1.0},
                "total_traces": total_traces,
                "correct_count": correct_count,
                "natural_length_distribution": {
                    str(length): int(length_distribution[length])
                    for length in sorted(length_distribution)
                },
            }
        )
    return rows, missing_question_ids


def build_per_question_lcurve_rows(
    *,
    question_id: str,
    traces: list[dict[str, Any]],
    min_length: int,
    min_bin_size: int,
) -> list[dict[str, Any]]:
    """Build one question's exact-length L-curve rows."""

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_length[int(trace["actual_num_steps"])].append(trace)

    rows: list[dict[str, Any]] = []
    for length in sorted(by_length):
        if length < min_length:
            continue
        length_rows = by_length[length]
        if len(length_rows) < min_bin_size:
            continue
        outcomes = [1 if bool(trace["is_correct"]) else 0 for trace in length_rows]
        mean_accuracy = sum(outcomes) / len(outcomes)
        rows.append(
            {
                "question_id": question_id,
                "L": length,
                "n": len(outcomes),
                "accuracy": mean_accuracy,
                "accuracy_se": standard_error(outcomes),
            }
        )
    return rows


def build_lstar_payload(
    *,
    question_meta: dict[str, Any],
    lcurve_rows: list[dict[str, Any]],
    smoothing_window: int,
    min_lcurve_bins: int,
) -> dict[str, Any]:
    """Build one question's L* summary payload."""

    l_star_a = select_raw_l_star(lcurve_rows)
    l_star_s = select_smoothed_l_star(
        lcurve_rows,
        smoothing_window=smoothing_window,
    )
    valid_bin_count = len(lcurve_rows)
    l_curve_insufficient = bool(question_meta["degenerate"]) or valid_bin_count < min_lcurve_bins
    if l_curve_insufficient:
        l_star_a = None
        l_star_s = None

    return {
        "pipeline": PQ_PIPELINE_NAME,
        "question_id": str(question_meta["question_id"]),
        "difficulty": str(question_meta["difficulty"]),
        "acc_pq": float(question_meta["acc_pq"]),
        "difficulty_score": float(question_meta["difficulty_score"]),
        "degenerate": bool(question_meta["degenerate"]),
        "valid_lcurve_bins": valid_bin_count,
        "l_curve_insufficient": l_curve_insufficient,
        "l_star_A": l_star_a,
        "l_star_S": l_star_s,
        "l_star_consistent": (
            l_star_a is not None
            and l_star_s is not None
            and int(l_star_a) == int(l_star_s)
        ),
    }


def select_raw_l_star(rows: list[dict[str, Any]]) -> int | None:
    """Select the raw argmax L* with smallest-L tie breaking."""

    if not rows:
        return None
    best_row = max(
        rows,
        key=lambda row: (float(row["accuracy"]), -int(row["L"])),
    )
    return int(best_row["L"])


def select_smoothed_l_star(
    rows: list[dict[str, Any]],
    *,
    smoothing_window: int,
) -> int | None:
    """Select the smoothed argmax L* with boundary shrinkage."""

    if not rows:
        return None
    if smoothing_window != 3:
        raise ValueError("Per-question L*_S currently expects a smoothing window of 3.")

    accuracy_by_length = {
        int(row["L"]): float(row["accuracy"])
        for row in rows
    }
    smoothed_rows: list[tuple[int, float]] = []
    for length in sorted(accuracy_by_length):
        window = [
            accuracy_by_length[neighbor]
            for neighbor in (length - 1, length, length + 1)
            if neighbor in accuracy_by_length
        ]
        smoothed_rows.append((length, sum(window) / len(window)))

    best_length, _ = max(
        smoothed_rows,
        key=lambda item: (item[1], -item[0]),
    )
    return int(best_length)


def export_per_question_bins(
    *,
    question_dir: Path,
    question_id: str,
    difficulty: str,
    traces: list[dict[str, Any]],
    token_counter: Any,
    integer_perturbation_range: tuple[int, int],
    float_perturbation_range: tuple[float, float],
    enable_tier3_semantic_flip: bool,
    corruption_token_delta_max: int,
    corruption_retry_limit: int,
    corruption_seed: int,
    min_length: int,
    min_retained_traces: int,
    max_retained_traces: int,
) -> list[dict[str, Any]]:
    """Write per-question exact-length bins and return their summaries."""

    bins_dir = question_dir / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_length[int(trace["actual_num_steps"])].append(trace)

    summaries: list[dict[str, Any]] = []
    for length in sorted(by_length):
        if length < min_length:
            continue
        length_rows = by_length[length]
        correct_rows = [
            row for row in length_rows
            if bool(row["is_correct"])
        ]
        complete_bundles: list[dict[str, Any]] = []
        for trace in correct_rows:
            records_by_mode = build_corruption_records(
                [("root", trace)],
                token_counter=token_counter,
                token_delta_max=corruption_token_delta_max,
                retry_limit=corruption_retry_limit,
                selection=CorruptionSelectionConfig(seed=corruption_seed),
                integer_perturbation_range=integer_perturbation_range,
                float_perturbation_range=float_perturbation_range,
                enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            )
            bundle = build_trace_sample_bundle(
                trace=trace,
                difficulty=difficulty,
                corruption_rows=records_by_mode["all_steps"],
            )
            if bundle is not None:
                complete_bundles.append(bundle)

        ordered_bundles = sorted(
            complete_bundles,
            key=lambda bundle: (
                int(bundle["meta_payload"]["trace_tier"]),
                _stable_seed(
                    f"{question_id}:{length}:{bundle['meta_payload']['source_trace_id']}"
                ),
            ),
        )
        retained_bundles = (
            ordered_bundles[:max_retained_traces]
            if len(ordered_bundles) >= min_retained_traces
            else []
        )
        _assign_local_sample_ids(retained_bundles)

        bin_dir = bins_dir / f"bin_{length}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        selection_rows = [bundle["selection_row"] for bundle in retained_bundles]
        _write_jsonl(bin_dir / "selection.jsonl", selection_rows)

        if retained_bundles:
            samples_dir = bin_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for bundle in retained_bundles:
                sample_dir = samples_dir / str(bundle["sample_id"])
                sample_dir.mkdir(parents=True, exist_ok=True)
                _write_json(sample_dir / "clean.json", bundle["clean_payload"])
                _write_json(sample_dir / "meta.json", bundle["meta_payload"])
                for k, payload in bundle["corrupt_payloads"].items():
                    _write_json(sample_dir / f"corrupt_k{k}.json", payload)
                for k, payload in bundle["corrupt_full_payloads"].items():
                    _write_json(sample_dir / f"corrupt_k{k}_full.json", payload)

        summary = {
            "scope": question_id,
            "pipeline": PQ_PIPELINE_NAME,
            "difficulty": difficulty,
            "L": length,
            "n_total_traces": len(length_rows),
            "n_correct": len(correct_rows),
            "n_tier1": sum(
                1 for bundle in complete_bundles
                if int(bundle["meta_payload"]["trace_tier"]) == 1
            ),
            "n_tier2": sum(
                1 for bundle in complete_bundles
                if int(bundle["meta_payload"]["trace_tier"]) == 2
            ),
            "n_failed": len(correct_rows) - len(complete_bundles),
            "n_retained": len(retained_bundles),
            "bin_status": "ok" if retained_bundles else "insufficient",
        }
        _write_json(bin_dir / "bin_summary.json", summary)
        summaries.append(summary)

    return summaries


def _assign_local_sample_ids(bundles: list[dict[str, Any]]) -> None:
    """Assign deterministic sample ids local to one per-question length bin."""

    for index, bundle in enumerate(bundles, start=1):
        sample_id = str(index)
        bundle["sample_id"] = sample_id
        bundle["meta_payload"]["sample_id"] = sample_id
        bundle["selection_row"]["sample_id"] = sample_id
        bundle["selection_row"]["sample_path"] = f"samples/{sample_id}"


def _group_traces_by_question(
    traces: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace["question_id"])].append(trace)
    return grouped


def _write_lcurve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "L", "n", "accuracy", "accuracy_se"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "L": int(row["L"]),
                    "n": int(row["n"]),
                    "accuracy": f"{float(row['accuracy']):.6f}",
                    "accuracy_se": f"{float(row['accuracy_se']):.6f}",
                }
            )


def _coalesce_int(value: int | None, *, default: int) -> int:
    return default if value is None else int(value)


def _validate_positive_odd_window(window: int) -> None:
    if window <= 0 or window % 2 == 0:
        raise ValueError("Per-question smoothing window must be a positive odd integer.")


__all__ = [
    "PQ_PIPELINE_NAME",
    "QUESTION_METADATA_FILENAME",
    "PER_QUESTION_DIRNAME",
    "aggregate_per_question_outputs",
    "build_lstar_payload",
    "build_per_question_lcurve_rows",
    "build_per_question_metadata_rows",
    "select_raw_l_star",
    "select_smoothed_l_star",
]
