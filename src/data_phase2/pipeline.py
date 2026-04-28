"""Pipeline and IO helpers for Stage 1 data-phase artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.common.settings import ExperimentConfig, require_config_value
from src.data_phase2.coarse_analysis import (
    DIFFICULTY_ORDER,
    build_accuracy_rows_by_difficulty,
    build_question_metadata_v4,
)
from src.data_phase2.difficulty_groups import export_difficulty_length_groups
from src.data_phase2.difficulty_histogram import export_difficulty_histogram


def aggregate_stage1_outputs(
    run_dir: str,
    *,
    config_path: str = "configs/stage1.yaml",
    hard_accuracy_threshold: float | None = None,
    easy_accuracy_threshold: float | None = None,
) -> dict[str, Any]:
    """Aggregate Stage C outputs into summary tables and coarse-analysis artifacts."""

    config = ExperimentConfig.from_yaml(config_path)
    min_nldd_length = require_config_value(
        "analysis.min_nldd_length",
        config.analysis.min_nldd_length,
    )
    resolved_hard_accuracy_threshold = (
        hard_accuracy_threshold
        if hard_accuracy_threshold is not None
        else require_config_value(
            "analysis.hard_accuracy_threshold",
            config.analysis.hard_accuracy_threshold,
        )
    )
    resolved_easy_accuracy_threshold = (
        easy_accuracy_threshold
        if easy_accuracy_threshold is not None
        else require_config_value(
            "analysis.easy_accuracy_threshold",
            config.analysis.easy_accuracy_threshold,
        )
    )
    min_cell_size = require_config_value(
        "analysis.min_cell_size",
        config.analysis.min_cell_size,
    )
    run_path = Path(run_dir)
    traces_path = run_path / "traces.jsonl"
    traces = load_stage1_traces(run_path)
    if not traces:
        raise ValueError(f"Stage C requires at least one trace in '{traces_path}'.")

    question_metadata = build_question_metadata_v4(
        traces=traces,
        hard_accuracy_threshold=resolved_hard_accuracy_threshold,
        easy_accuracy_threshold=resolved_easy_accuracy_threshold,
    )
    accuracy_rows = build_accuracy_rows_by_difficulty(
        traces=traces,
        question_metadata=question_metadata,
        min_nldd_length=min_nldd_length,
    )

    accuracy_path = run_path / "accuracy_by_length.csv"
    metadata_path = run_path / "question_metadata.jsonl"
    difficulty_histogram_path = run_path / "difficulty_histogram.csv"
    _write_accuracy_rows(accuracy_path, accuracy_rows)
    _write_question_metadata(metadata_path, question_metadata)
    export_difficulty_histogram(
        question_metadata_path=metadata_path,
        output_path=difficulty_histogram_path,
    )
    difficulty_export = export_difficulty_length_groups(
        run_path,
        config_path=config_path,
    )

    return {
        "accuracy_by_length_path": str(accuracy_path),
        "question_metadata_path": str(metadata_path),
        "difficulty_histogram_path": str(difficulty_histogram_path),
        "difficulty_root_path": str(run_path / "difficulty"),
        "difficulty_question_counts": {
            difficulty: difficulty_export["difficulties"][difficulty]["question_count"]
            for difficulty in DIFFICULTY_ORDER
        },
        "difficulty_trace_counts": {
            difficulty: difficulty_export["difficulties"][difficulty]["trace_count"]
            for difficulty in DIFFICULTY_ORDER
        },
        "difficulty_sample_counts": {
            difficulty: difficulty_export["difficulties"][difficulty]["selected_sample_count"]
            for difficulty in DIFFICULTY_ORDER
        },
        "num_questions": len(question_metadata),
        "num_traces": len(traces),
        "num_deduped_traces": None,
        "hard_accuracy_threshold": resolved_hard_accuracy_threshold,
        "easy_accuracy_threshold": resolved_easy_accuracy_threshold,
    }


def load_stage1_traces(run_path: Path) -> list[dict[str, Any]]:
    """Load formal traces from the run root or materialize them from shard outputs."""

    traces_path = run_path / "traces.jsonl"
    if traces_path.exists():
        return _load_jsonl_records(traces_path)

    shard_paths = discover_stage1_shard_paths(run_path)
    if shard_paths:
        traces = merge_stage1_shards(shard_paths)
        _write_jsonl_records(traces_path, traces)
        ensure_root_run_metadata(run_path, shard_paths)
        return traces

    difficulty_trace_paths = discover_exported_difficulty_trace_paths(run_path)
    if difficulty_trace_paths:
        traces = merge_stage1_trace_files(difficulty_trace_paths)
        _write_jsonl_records(traces_path, traces)
        return traces

    raise FileNotFoundError(
        f"Stage E requires traces.jsonl at '{traces_path}', shard traces under "
        f"'{run_path / 'shards'}', or per-difficulty traces under "
        f"'{run_path / 'difficulty'}'."
    )


def discover_stage1_shard_paths(run_path: Path) -> list[Path]:
    """Discover shard trace files in a shared run directory."""

    shards_root = run_path / "shards"
    if not shards_root.exists():
        return []
    return sorted(path for path in shards_root.glob("*/traces.jsonl") if path.is_file())


def discover_exported_difficulty_trace_paths(run_path: Path) -> list[Path]:
    """Discover per-difficulty trace tables from an exported data-phase handoff."""

    difficulty_root = run_path / "difficulty"
    if not difficulty_root.exists():
        return []
    return sorted(
        path
        for path in difficulty_root.glob("*/traces.jsonl")
        if path.is_file()
    )


def merge_stage1_shards(shard_paths: list[Path]) -> list[dict[str, Any]]:
    """Merge shard trace files, keeping the first occurrence of each trace id."""

    return merge_stage1_trace_files(shard_paths)


def merge_stage1_trace_files(trace_paths: list[Path]) -> list[dict[str, Any]]:
    """Merge trace files, keeping the first occurrence of each trace id."""

    merged: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()

    for trace_path in trace_paths:
        for record in _load_jsonl_records(trace_path):
            trace_id = record.get("trace_id")
            if isinstance(trace_id, str):
                if trace_id in seen_trace_ids:
                    continue
                seen_trace_ids.add(trace_id)
            merged.append(record)

    return merged


def ensure_root_run_metadata(run_path: Path, shard_paths: list[Path]) -> None:
    """Copy canonical run metadata from the first shard when the root metadata is absent."""

    root_meta_path = run_path / "run_meta.json"
    if root_meta_path.exists():
        return

    for shard_path in shard_paths:
        shard_meta_path = shard_path.with_name("run_meta.json")
        if shard_meta_path.exists():
            root_meta_path.write_text(
                shard_meta_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return


def plot_stage1_figures(run_dir: str) -> None:
    """Create Stage 1 figures from aggregated outputs."""

    raise NotImplementedError("Plotting is not implemented yet.")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_accuracy_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["difficulty", "length", "n", "mean_accuracy", "se_accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "difficulty": row["difficulty"],
                    "length": int(row["length"]),
                    "n": int(row["n"]),
                    "mean_accuracy": f"{float(row['mean_accuracy']):.6f}",
                    "se_accuracy": f"{float(row['se_accuracy']):.6f}",
                }
            )


def _write_question_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
