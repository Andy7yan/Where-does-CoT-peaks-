"""Exact-length aggregation helpers for the current Stage 1 data phase."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_accuracy_rows(
    traces: list[dict[str, Any]],
    *,
    metadata_by_question: dict[str, dict[str, Any]],
    difficulty: str,
    min_nldd_length: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate exact-length accuracy rows for one difficulty bucket."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for trace in traces:
        question_meta = metadata_by_question.get(str(trace["question_id"]), {})
        if question_meta.get("difficulty_bucket") != difficulty:
            continue
        length = int(trace["actual_num_steps"])
        if min_nldd_length is not None and length < min_nldd_length:
            continue
        grouped[length].append(1 if trace["is_correct"] else 0)

    rows: list[dict[str, Any]] = []
    for length in sorted(grouped):
        outcomes = grouped[length]
        mean_accuracy = sum(outcomes) / len(outcomes)
        rows.append(
            {
                "difficulty": difficulty,
                "length": length,
                "n": len(outcomes),
                "mean_accuracy": mean_accuracy,
                "se_accuracy": standard_error(outcomes),
            }
        )
    return rows


def select_l_star_from_accuracy_rows(rows: list[dict[str, Any]]) -> int | None:
    """Resolve L* as the smallest exact length with maximal mean accuracy."""

    if not rows:
        return None
    best_row = max(
        rows,
        key=lambda row: (float(row["mean_accuracy"]), -int(row["length"])),
    )
    return int(best_row["length"])


def standard_error(outcomes: list[int]) -> float:
    """Compute Bernoulli standard error for a binary outcome list."""

    if not outcomes:
        return 0.0
    mean = sum(outcomes) / len(outcomes)
    return (mean * (1.0 - mean) / len(outcomes)) ** 0.5


__all__ = [
    "build_accuracy_rows",
    "select_l_star_from_accuracy_rows",
    "standard_error",
]
