"""Reporting helpers for aggregation and plotting outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.settings import ExperimentConfig, require_config_value


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
    """Aggregate Stage E accuracy outputs into summary tables."""

    config = ExperimentConfig.from_yaml(config_path)
    min_bin_size = require_config_value(
        "analysis.min_bin_size",
        config.analysis.min_bin_size,
    )
    run_path = Path(run_dir)
    traces_path = run_path / "traces.jsonl"
    traces = load_stage1_traces(run_path)
    if not traces:
        raise ValueError(f"Stage E requires at least one trace in '{traces_path}'.")

    buckets = build_accuracy_buckets(traces, min_bin_size=min_bin_size)
    if not buckets:
        raise ValueError("Stage E failed to build any accuracy buckets from traces.")
    l_star = select_l_star(buckets)
    question_metadata = build_question_metadata(traces)

    accuracy_path = run_path / "accuracy_by_length.csv"
    metadata_path = run_path / "question_metadata.jsonl"
    _write_accuracy_rows(accuracy_path, buckets)
    _write_question_metadata(metadata_path, question_metadata)

    return {
        "accuracy_by_length_path": str(accuracy_path),
        "question_metadata_path": str(metadata_path),
        "l_star": l_star,
        "num_buckets": len(buckets),
        "num_questions": len(question_metadata),
        "num_traces": len(traces),
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


def select_l_star(buckets: list[AccuracyBucket]) -> float:
    """Select the smallest bucket label among the highest-accuracy buckets."""

    best_bucket = min(
        buckets,
        key=lambda bucket: (-bucket.mean, bucket.bucket_label),
    )
    return best_bucket.bucket_label


def build_question_metadata(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-question metadata required by Stage E downstream steps."""

    traces_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        traces_by_question[str(trace["question_id"])].append(trace)

    metadata_rows: list[dict[str, Any]] = []
    for question_id in sorted(traces_by_question):
        question_traces = traces_by_question[question_id]
        total_samples = len(question_traces)
        correct_count = sum(1 for trace in question_traces if trace["is_correct"])
        accuracy = (correct_count / total_samples) if total_samples else 0.0
        length_distribution = Counter(
            int(trace["actual_num_steps"]) for trace in question_traces
        )
        grouped_accuracy: dict[int, list[int]] = defaultdict(list)
        for trace in question_traces:
            grouped_accuracy[int(trace["actual_num_steps"])].append(1 if trace["is_correct"] else 0)

        optimal_length = None
        if grouped_accuracy:
            optimal_length = min(
                grouped_accuracy,
                key=lambda length: (
                    -sum(grouped_accuracy[length]) / len(grouped_accuracy[length]),
                    length,
                ),
            )

        metadata_rows.append(
            {
                "question_id": question_id,
                "difficulty": 1.0 - accuracy,
                "accuracy": accuracy,
                "optimal_length": optimal_length,
                "total_samples": total_samples,
                "correct_count": correct_count,
                "natural_length_distribution": {
                    str(length): length_distribution[length]
                    for length in sorted(length_distribution)
                },
            }
        )

    return metadata_rows


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


def _write_accuracy_rows(path: Path, buckets: list[AccuracyBucket]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket_label", "n", "mean", "se"])
        writer.writeheader()
        for bucket in buckets:
            writer.writerow(
                {
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


def _format_bucket_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"
