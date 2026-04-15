"""Reporting helpers for aggregation and plotting outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.data_phase2.coarse_analysis import (
    DIFFICULTY_ORDER,
    build_accuracy_buckets_by_difficulty,
    build_coarse_analysis,
    build_question_metadata_v4,
    dedupe_traces_for_analysis,
)
from src.common.settings import ExperimentConfig, require_config_value


@dataclass
class AccuracyBucket:
    """An accuracy bucket after optional min-size merging."""

    lengths: list[int]
    outcomes: list[int]

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def mean(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(self.outcomes) / len(self.outcomes)

    @property
    def se(self) -> float:
        if not self.outcomes:
            return 0.0
        mean = self.mean
        return math.sqrt(mean * (1.0 - mean) / len(self.outcomes))

    @property
    def bucket_label(self) -> float:
        sorted_lengths = sorted(self.lengths)
        midpoint = len(sorted_lengths) // 2
        if len(sorted_lengths) % 2 == 1:
            return float(sorted_lengths[midpoint])
        return (sorted_lengths[midpoint - 1] + sorted_lengths[midpoint]) / 2.0


def aggregate_stage1_outputs(
    run_dir: str,
    *,
    config_path: str = "configs/stage1.yaml",
) -> dict[str, Any]:
    """Aggregate Stage C outputs into summary tables and coarse-analysis artifacts."""

    config = ExperimentConfig.from_yaml(config_path)
    min_bin_size = require_config_value(
        "analysis.min_bin_size",
        config.analysis.min_bin_size,
    )
    min_nldd_length = require_config_value(
        "analysis.min_nldd_length",
        config.analysis.min_nldd_length,
    )
    difficulty_quantiles = require_config_value(
        "analysis.difficulty_quantiles",
        config.analysis.difficulty_quantiles,
    )
    primary_lstar_window = require_config_value(
        "analysis.primary_lstar_window",
        config.analysis.primary_lstar_window,
    )
    fallback_lstar_window = require_config_value(
        "analysis.fallback_lstar_window",
        config.analysis.fallback_lstar_window,
    )
    min_near_lstar_traces = require_config_value(
        "analysis.min_near_lstar_traces",
        config.analysis.min_near_lstar_traces,
    )
    min_cell_size = require_config_value(
        "analysis.min_cell_size",
        config.analysis.min_cell_size,
    )
    accuracy_exclusion_bounds = require_config_value(
        "analysis.accuracy_exclusion_bounds",
        config.analysis.accuracy_exclusion_bounds,
    )
    run_path = Path(run_dir)
    traces_path = run_path / "traces.jsonl"
    traces = load_stage1_traces(run_path)
    if not traces:
        raise ValueError(f"Stage C requires at least one trace in '{traces_path}'.")

    deduped_traces = dedupe_traces_for_analysis(traces)
    question_metadata = build_question_metadata_v4(
        traces=traces,
        deduped_traces=deduped_traces,
        difficulty_quantiles=difficulty_quantiles,
        accuracy_exclusion_bounds=accuracy_exclusion_bounds,
    )
    buckets_by_difficulty = build_accuracy_buckets_by_difficulty(
        traces=traces,
        deduped_traces=deduped_traces,
        question_metadata=question_metadata,
        build_accuracy_buckets=build_accuracy_buckets,
        min_bin_size=min_bin_size,
    )
    coarse_analysis = build_coarse_analysis(
        deduped_traces=deduped_traces,
        question_metadata=question_metadata,
        buckets_by_difficulty=buckets_by_difficulty,
        min_nldd_length=min_nldd_length,
        primary_lstar_window=primary_lstar_window,
        fallback_lstar_window=fallback_lstar_window,
        min_near_lstar_traces=min_near_lstar_traces,
        min_cell_size=min_cell_size,
        select_l_star=select_l_star,
    )

    accuracy_path = run_path / "accuracy_by_length.csv"
    metadata_path = run_path / "question_metadata.jsonl"
    coarse_analysis_path = run_path / "coarse_analysis.json"
    lstar_summary_path = run_path / "lstar_summary.csv"
    _write_accuracy_rows(accuracy_path, buckets_by_difficulty)
    _write_question_metadata(metadata_path, question_metadata)
    coarse_analysis_path.write_text(
        json.dumps(coarse_analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_lstar_summary(lstar_summary_path, coarse_analysis)

    return {
        "accuracy_by_length_path": str(accuracy_path),
        "question_metadata_path": str(metadata_path),
        "coarse_analysis_path": str(coarse_analysis_path),
        "lstar_summary_path": str(lstar_summary_path),
        "difficulty_question_counts": {
            difficulty: coarse_analysis["difficulties"][difficulty]["question_count"]
            for difficulty in DIFFICULTY_ORDER
        },
        "num_questions": len(question_metadata),
        "num_traces": len(traces),
        "num_deduped_traces": len(deduped_traces),
    }


def load_stage1_traces(run_path: Path) -> list[dict[str, Any]]:
    """Load formal traces from the run root or materialize them from shard outputs."""

    traces_path = run_path / "traces.jsonl"
    if traces_path.exists():
        return _load_jsonl_records(traces_path)

    shard_paths = discover_stage1_shard_paths(run_path)
    if not shard_paths:
        raise FileNotFoundError(
            f"Stage E requires traces.jsonl at '{traces_path}' or shard traces under "
            f"'{run_path / 'shards'}'."
        )

    traces = merge_stage1_shards(shard_paths)
    _write_jsonl_records(traces_path, traces)
    ensure_root_run_metadata(run_path, shard_paths)
    return traces


def discover_stage1_shard_paths(run_path: Path) -> list[Path]:
    """Discover shard trace files in a shared run directory."""

    shards_root = run_path / "shards"
    if not shards_root.exists():
        return []
    return sorted(path for path in shards_root.glob("*/traces.jsonl") if path.is_file())


def merge_stage1_shards(shard_paths: list[Path]) -> list[dict[str, Any]]:
    """Merge shard trace files, keeping the first occurrence of each trace id."""

    merged: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()

    for shard_path in shard_paths:
        for record in _load_jsonl_records(shard_path):
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


def build_accuracy_buckets(
    traces: list[dict[str, Any]],
    *,
    min_bin_size: int,
) -> list[AccuracyBucket]:
    """Group traces by effective length and merge sparse bins."""

    if min_bin_size <= 0:
        raise ValueError("analysis.min_bin_size must be positive.")

    grouped: dict[int, list[int]] = defaultdict(list)
    for trace in traces:
        length = int(trace["actual_num_steps"])
        grouped[length].append(1 if trace["is_correct"] else 0)

    buckets = [
        AccuracyBucket(
            lengths=[length] * len(grouped[length]),
            outcomes=list(grouped[length]),
        )
        for length in sorted(grouped)
    ]
    return merge_sparse_accuracy_buckets(buckets, min_bin_size=min_bin_size)


def merge_sparse_accuracy_buckets(
    buckets: list[AccuracyBucket],
    *,
    min_bin_size: int,
) -> list[AccuracyBucket]:
    """Merge sparse neighboring buckets until every remaining bucket is large enough."""

    merged = [AccuracyBucket(lengths=list(bucket.lengths), outcomes=list(bucket.outcomes)) for bucket in buckets]
    if len(merged) <= 1:
        return merged

    while True:
        sparse_index = next(
            (index for index, bucket in enumerate(merged) if bucket.n < min_bin_size),
            None,
        )
        if sparse_index is None or len(merged) == 1:
            break

        target_index = choose_merge_neighbor(merged, sparse_index)
        left_index, right_index = sorted((sparse_index, target_index))
        combined = AccuracyBucket(
            lengths=sorted(merged[left_index].lengths + merged[right_index].lengths),
            outcomes=merged[left_index].outcomes + merged[right_index].outcomes,
        )
        merged[left_index:right_index + 1] = [combined]

    return merged


def choose_merge_neighbor(buckets: list[AccuracyBucket], sparse_index: int) -> int:
    """Pick the adjacent bucket used to absorb a sparse bucket."""

    if sparse_index < 0 or sparse_index >= len(buckets):
        raise IndexError("Sparse bucket index is out of range.")
    if len(buckets) == 1:
        raise ValueError("Cannot merge when only one accuracy bucket exists.")

    if sparse_index == 0:
        return 1
    if sparse_index == len(buckets) - 1:
        return len(buckets) - 2

    left_bucket = buckets[sparse_index - 1]
    right_bucket = buckets[sparse_index + 1]
    if right_bucket.n > left_bucket.n:
        return sparse_index + 1
    return sparse_index - 1


def select_l_star(buckets: list[AccuracyBucket]) -> float | None:
    """Select the smallest bucket label among the highest-accuracy buckets."""

    if not buckets:
        return None
    best_bucket = min(
        buckets,
        key=lambda bucket: (-bucket.mean, bucket.bucket_label),
    )
    return best_bucket.bucket_label


def build_question_metadata(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for callers that still expect metadata rows."""

    return build_question_metadata_v4(
        traces=traces,
        deduped_traces=dedupe_traces_for_analysis(traces),
        difficulty_quantiles=[0.22, 0.55],
        accuracy_exclusion_bounds=[0.01, 0.99],
    )


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


def _write_accuracy_rows(
    path: Path,
    buckets_by_difficulty: dict[str, dict[str, list[AccuracyBucket]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["difficulty", "dedup_mode", "bucket_label", "n", "mean", "se"],
        )
        writer.writeheader()
        for difficulty in DIFFICULTY_ORDER:
            for dedup_mode in ("dedup", "raw"):
                for bucket in buckets_by_difficulty[difficulty][dedup_mode]:
                    writer.writerow(
                        {
                            "difficulty": difficulty,
                            "dedup_mode": dedup_mode,
                            "bucket_label": _format_bucket_label(bucket.bucket_label),
                            "n": bucket.n,
                            "mean": f"{bucket.mean:.6f}",
                            "se": f"{bucket.se:.6f}",
                        }
                    )


def _write_question_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lstar_summary(path: Path, coarse_analysis: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "difficulty",
                "L_star",
                "window_left",
                "window_right",
                "n_traces_near_lstar",
                "window_status",
            ],
        )
        writer.writeheader()
        for difficulty in DIFFICULTY_ORDER:
            row = coarse_analysis["difficulties"][difficulty]
            selected_window = row["near_lstar"]["selected_window"]
            writer.writerow(
                {
                    "difficulty": difficulty,
                    "L_star": "" if row["l_star"] is None else _format_bucket_label(float(row["l_star"])),
                    "window_left": "" if selected_window is None else _format_bucket_label(float(selected_window[0])),
                    "window_right": "" if selected_window is None else _format_bucket_label(float(selected_window[1])),
                    "n_traces_near_lstar": row["near_lstar"]["selected_count"],
                    "window_status": row["near_lstar"]["status"],
                }
            )


def _format_bucket_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"
