"""Accuracy-bucket helpers for Stage 1 data-phase aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

from src.data_phase2.coarse_analysis import build_question_metadata_v4, dedupe_traces_for_analysis


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


def build_question_metadata(
    traces: list[dict[str, Any]],
    *,
    hard_accuracy_threshold: float = 0.5,
    easy_accuracy_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for callers that still expect metadata rows."""

    return build_question_metadata_v4(
        traces=traces,
        deduped_traces=dedupe_traces_for_analysis(traces),
        hard_accuracy_threshold=hard_accuracy_threshold,
        easy_accuracy_threshold=easy_accuracy_threshold,
    )


def _format_bucket_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"
