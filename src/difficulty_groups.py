"""Export per-difficulty jsonl groups for the canonical full-generation run."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from src.coarse_analysis import DIFFICULTY_ORDER, dedupe_traces_for_analysis, interpolated_quantile
from src.reports import load_stage1_traces


LENGTH_GROUP_ORDER = ("short", "medium", "detailed", "verbose")
LT3_GROUP_NAME = "lt3_excluded"


def export_difficulty_length_groups(run_path: Path) -> dict[str, Any]:
    """Write grouped trace exports under `run_path/difficulty_groups`."""

    traces = load_stage1_traces(run_path)
    metadata_by_question = {
        str(row["question_id"]): row
        for row in _load_jsonl(run_path / "question_metadata.jsonl")
    }
    coarse_analysis = json.loads((run_path / "coarse_analysis.json").read_text(encoding="utf-8"))
    min_nldd_length = int(coarse_analysis.get("notes", {}).get("min_nldd_length") or 3)

    output_dir = run_path / "difficulty_groups"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _compute_thresholds(
        traces=traces,
        metadata_by_question=metadata_by_question,
        min_nldd_length=min_nldd_length,
    )
    grouped_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        difficulty: {
            "all": [],
            LT3_GROUP_NAME: [],
            **{label: [] for label in LENGTH_GROUP_ORDER},
        }
        for difficulty in DIFFICULTY_ORDER
    }
    excluded_rows: list[dict[str, Any]] = []

    sorted_traces = sorted(
        traces,
        key=lambda row: (
            str(row.get("question_id", "")),
            int(row.get("actual_num_steps", 0)),
            str(row.get("prompt_id", "")),
            str(row.get("trace_id", "")),
        ),
    )
    for trace in sorted_traces:
        question_id = str(trace["question_id"])
        question_meta = metadata_by_question.get(question_id, {})
        difficulty = question_meta.get("difficulty_bucket")
        enriched = dict(trace)
        enriched["difficulty_bucket"] = difficulty
        enriched["excluded_from_difficulty"] = bool(question_meta.get("excluded_from_difficulty", True))
        enriched["nldd_min_length"] = min_nldd_length
        enriched["nldd_length_eligible"] = int(trace["actual_num_steps"]) >= min_nldd_length
        enriched["nldd_measurement_eligible"] = (
            bool(trace.get("is_correct")) and int(trace["actual_num_steps"]) >= min_nldd_length
        )

        if difficulty not in DIFFICULTY_ORDER:
            enriched["difficulty_length_group"] = None
            excluded_rows.append(enriched)
            continue

        group_name = _assign_length_group(
            length=int(trace["actual_num_steps"]),
            min_nldd_length=min_nldd_length,
            threshold_row=thresholds[difficulty],
        )
        threshold_row = thresholds[difficulty]
        enriched["difficulty_length_group"] = group_name
        enriched["length_group_q25"] = threshold_row["q25"]
        enriched["length_group_q50"] = threshold_row["q50"]
        enriched["length_group_q75"] = threshold_row["q75"]

        grouped_rows[difficulty]["all"].append(enriched)
        grouped_rows[difficulty][group_name].append(enriched)

    manifest = {
        "schema_version": "stage1_difficulty_groups_v1",
        "min_nldd_length": min_nldd_length,
        "difficulty_groups": {},
        "excluded_from_difficulty_count": len(excluded_rows),
    }
    for difficulty in DIFFICULTY_ORDER:
        difficulty_dir = output_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for name, rows in grouped_rows[difficulty].items():
            _write_jsonl(difficulty_dir / f"{name}.jsonl", rows)
            counts[name] = len(rows)
        manifest["difficulty_groups"][difficulty] = {
            "thresholds": thresholds[difficulty],
            "counts": counts,
        }

    _write_jsonl(output_dir / "excluded_from_difficulty.jsonl", excluded_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _compute_thresholds(
    *,
    traces: list[dict[str, Any]],
    metadata_by_question: dict[str, dict[str, Any]],
    min_nldd_length: int,
) -> dict[str, dict[str, float | None]]:
    deduped_traces = dedupe_traces_for_analysis(traces)
    rows: dict[str, dict[str, float | None]] = {}
    for difficulty in DIFFICULTY_ORDER:
        lengths = [
            int(trace["actual_num_steps"])
            for trace in deduped_traces
            if trace.get("is_correct")
            and int(trace["actual_num_steps"]) >= min_nldd_length
            and metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
        if not lengths:
            rows[difficulty] = {"q25": None, "q50": None, "q75": None}
            continue
        rows[difficulty] = {
            "q25": interpolated_quantile(lengths, 0.25),
            "q50": interpolated_quantile(lengths, 0.50),
            "q75": interpolated_quantile(lengths, 0.75),
        }
    return rows


def _assign_length_group(
    *,
    length: int,
    min_nldd_length: int,
    threshold_row: dict[str, float | None],
) -> str:
    if length < min_nldd_length:
        return LT3_GROUP_NAME

    q25 = threshold_row["q25"]
    q50 = threshold_row["q50"]
    q75 = threshold_row["q75"]
    if q25 is None or q50 is None or q75 is None:
        return "verbose"
    if length <= q25:
        return "short"
    if length <= q50:
        return "medium"
    if length <= q75:
        return "detailed"
    return "verbose"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
