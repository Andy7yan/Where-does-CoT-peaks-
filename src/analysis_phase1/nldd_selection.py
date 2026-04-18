"""Trace-selection helpers for the NLDD full-run workflow."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER, assign_length_bin, dedupe_traces_for_analysis

from src.analysis_phase1.nldd_shared import (
    TRACE_SELECTION_FIELDNAMES,
    TRACE_SELECTION_REQUIRED_COLUMNS,
    _load_jsonl_records,
    _normalize_optional_string,
    _parse_bool,
    _stable_seed,
)


@dataclass(frozen=True)
class TraceSelectionConfig:
    """Selection settings for the v4 full-run NLDD trace sweep."""

    target_traces_per_cell: int
    target_traces_near_lstar: int
    per_question_trace_cap: int
    min_nldd_length: int
    seed: int = 42


def build_v4_trace_selection(
    *,
    traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    coarse_analysis: dict[str, Any],
    selection_config: TraceSelectionConfig,
) -> list[dict[str, Any]]:
    """Build the v4 full-run trace selection table from Stage C artifacts."""

    metadata_by_question = {str(row["question_id"]): row for row in question_metadata}
    deduped_traces = dedupe_traces_for_analysis(traces)
    candidate_rows: list[dict[str, Any]] = []
    for trace in deduped_traces:
        if not trace.get("is_correct"):
            continue
        actual_length = int(trace.get("actual_num_steps", 0))
        if actual_length < selection_config.min_nldd_length:
            continue
        question_meta = metadata_by_question.get(str(trace["question_id"]))
        if not question_meta:
            continue
        difficulty = question_meta.get("difficulty_bucket")
        if difficulty not in DIFFICULTY_ORDER:
            continue

        difficulty_payload = coarse_analysis.get("difficulties", {}).get(difficulty, {})
        tertiles = difficulty_payload.get("length_tertiles", {})
        q33 = tertiles.get("q33")
        q67 = tertiles.get("q67")
        if q33 is None or q67 is None:
            continue

        raw_length_bin = assign_length_bin(actual_length, float(q33), float(q67))
        length_bin_map = difficulty_payload.get("length_bin_map", {})
        length_bin = length_bin_map.get(raw_length_bin) or raw_length_bin
        candidate_rows.append(
            {
                "trace_id": str(trace["trace_id"]),
                "question_id": str(trace["question_id"]),
                "difficulty": str(difficulty),
                "length_bin": str(length_bin),
                "raw_length_bin": str(raw_length_bin),
                "actual_clean_length": actual_length,
                "prompt_id": str(trace.get("prompt_id", "")),
                "selected_for_nldd": False,
                "selected_for_near_lstar": False,
                "selection_mode": None,
                "near_lstar_selection_mode": None,
            }
        )

    if not candidate_rows:
        return []

    rows_by_trace_id = {row["trace_id"]: row for row in candidate_rows}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_cell[(row["difficulty"], row["length_bin"])].append(row)

    for difficulty, difficulty_payload in coarse_analysis.get("difficulties", {}).items():
        for entry in difficulty_payload.get("merged_length_bins", []):
            label = entry.get("label")
            if not isinstance(label, str):
                continue
            selected_rows, selection_mode = _select_rows_for_measurement(
                rows=by_cell.get((difficulty, label), []),
                target=selection_config.target_traces_per_cell,
                per_question_cap=selection_config.per_question_trace_cap,
                seed=selection_config.seed,
                salt=f"cell:{difficulty}:{label}",
            )
            for row in selected_rows:
                stored = rows_by_trace_id[row["trace_id"]]
                stored["selected_for_nldd"] = True
                stored["selection_mode"] = selection_mode

    for difficulty, difficulty_payload in coarse_analysis.get("difficulties", {}).items():
        selected_window = difficulty_payload.get("near_lstar", {}).get("selected_window")
        if not isinstance(selected_window, list) or len(selected_window) != 2:
            continue
        left, right = float(selected_window[0]), float(selected_window[1])
        difficulty_rows = [
            row
            for row in candidate_rows
            if row["difficulty"] == difficulty and left <= row["actual_clean_length"] <= right
        ]
        preferred_ids = {row["trace_id"] for row in difficulty_rows if row["selected_for_nldd"]}
        selected_rows, selection_mode = _select_rows_for_measurement(
            rows=difficulty_rows,
            target=selection_config.target_traces_near_lstar,
            per_question_cap=selection_config.per_question_trace_cap,
            seed=selection_config.seed,
            salt=f"near_lstar:{difficulty}:{left}:{right}",
            preferred_trace_ids=preferred_ids,
        )
        for row in selected_rows:
            stored = rows_by_trace_id[row["trace_id"]]
            stored["selected_for_near_lstar"] = True
            stored["near_lstar_selection_mode"] = selection_mode

    selected_union = [
        row
        for row in rows_by_trace_id.values()
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    ]
    return sorted(
        selected_union,
        key=lambda row: (
            DIFFICULTY_ORDER.index(row["difficulty"]),
            row["length_bin"],
            row["actual_clean_length"],
            row["question_id"],
            row["trace_id"],
        ),
    )


def load_trace_selection(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the v4 trace-selection CSV."""

    selection_path = Path(path)
    if not selection_path.exists():
        raise FileNotFoundError(f"Trace selection file not found: {selection_path}")

    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            name
            for name in TRACE_SELECTION_REQUIRED_COLUMNS
            if name not in fieldnames
        ]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"trace_selection.csv is missing required columns: {joined}")

        rows: list[dict[str, Any]] = []
        seen_trace_ids: set[str] = set()
        for raw_row in reader:
            row = dict(raw_row)
            row["trace_id"] = str(row["trace_id"])
            if row["trace_id"] in seen_trace_ids:
                raise ValueError(f"trace_selection.csv contains duplicate trace_id: {row['trace_id']}")
            seen_trace_ids.add(row["trace_id"])
            row["question_id"] = str(row["question_id"])
            row["difficulty"] = str(row["difficulty"])
            row["length_bin"] = str(row["length_bin"])
            row["raw_length_bin"] = _normalize_optional_string(row.get("raw_length_bin"))
            row["prompt_id"] = str(row.get("prompt_id", ""))
            row["selection_mode"] = _normalize_optional_string(row.get("selection_mode"))
            row["near_lstar_selection_mode"] = _normalize_optional_string(
                row.get("near_lstar_selection_mode")
            )
            row["selected_for_nldd"] = _parse_bool(row["selected_for_nldd"])
            row["selected_for_near_lstar"] = _parse_bool(row["selected_for_near_lstar"])
            row["actual_clean_length"] = int(row.get("actual_clean_length") or 0)
            rows.append(row)

    if not rows:
        raise ValueError("trace_selection.csv is empty.")
    if not any(row["selected_for_nldd"] or row["selected_for_near_lstar"] for row in rows):
        raise ValueError("trace_selection.csv does not contain any selected traces.")
    return rows


def write_trace_selection(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the v4 trace-selection CSV."""

    selection_path = Path(path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_SELECTION_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            serialized = {name: row.get(name) for name in TRACE_SELECTION_FIELDNAMES}
            serialized["selected_for_nldd"] = "true" if row.get("selected_for_nldd") else "false"
            serialized["selected_for_near_lstar"] = (
                "true" if row.get("selected_for_near_lstar") else "false"
            )
            writer.writerow(serialized)


def load_question_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load the Stage C question metadata JSONL."""

    return _load_jsonl_records(Path(path))


def load_coarse_analysis(path: str | Path) -> dict[str, Any]:
    """Load the Stage C coarse-analysis JSON."""

    coarse_path = Path(path)
    with coarse_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("coarse_analysis.json must contain a top-level mapping.")
    return data


def load_or_build_trace_selection(
    *,
    run_dir: str,
    traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    coarse_analysis: dict[str, Any],
    selection_config: TraceSelectionConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Load trace_selection.csv if present, otherwise build the v4 selection."""

    run_path = Path(run_dir)
    selection_path = run_path / "trace_selection.csv"
    if selection_path.exists():
        return load_trace_selection(selection_path), "loaded"

    rows = build_v4_trace_selection(
        traces=traces,
        question_metadata=question_metadata,
        coarse_analysis=coarse_analysis,
        selection_config=selection_config,
    )
    if not rows:
        raise ValueError("Could not build a non-empty v4 trace selection from the available artifacts.")
    write_trace_selection(selection_path, rows)
    return rows, "built"


def _select_rows_for_measurement(
    *,
    rows: list[dict[str, Any]],
    target: int,
    per_question_cap: int,
    seed: int,
    salt: str,
    preferred_trace_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return [], "empty"
    if len(rows) <= target:
        return list(rows), "all_available"

    preferred_trace_ids = preferred_trace_ids or set()
    ordered_rows = sorted(
        rows,
        key=lambda row: _stable_seed(f"{seed}:{salt}:{row['trace_id']}"),
    )
    preferred_rows = [row for row in ordered_rows if row["trace_id"] in preferred_trace_ids]
    fallback_rows = [row for row in ordered_rows if row["trace_id"] not in preferred_trace_ids]

    selected: list[dict[str, Any]] = []
    question_counts: Counter[str] = Counter()
    for row in [*preferred_rows, *fallback_rows]:
        question_id = str(row["question_id"])
        if question_counts[question_id] >= per_question_cap:
            continue
        selected.append(row)
        question_counts[question_id] += 1
        if len(selected) >= target:
            break

    if len(selected) >= target:
        if preferred_rows:
            return selected, "preferred_reuse"
        if per_question_cap < len(rows):
            return selected, "capped_random_sample"
        return selected, "random_sample"

    selected_trace_ids = {row["trace_id"] for row in selected}
    relaxed_rows = [
        row
        for row in ordered_rows
        if row["trace_id"] not in selected_trace_ids
    ]
    return [*selected, *relaxed_rows], "all_after_cap"
