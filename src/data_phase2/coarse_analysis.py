"""Difficulty grouping helpers for the current Stage 1 data phase."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


DIFFICULTY_ORDER = ("easy", "medium", "hard")


def dedupe_traces_for_analysis(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate traces following the v4 `(question, L, steps)` rule."""

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, tuple[str, ...]]] = set()
    for trace in traces:
        key = (
            str(trace["question_id"]),
            int(trace["actual_num_steps"]),
            tuple(str(step) for step in trace.get("steps", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trace)
    return deduped


def assign_length_bin(actual_length: int, q33: float, q67: float) -> str:
    """Map a trace length to the canonical short/medium/long tertile label."""

    if actual_length <= q33:
        return "short"
    if actual_length <= q67:
        return "medium"
    return "long"


def build_question_metadata_v4(
    *,
    traces: list[dict[str, Any]],
    deduped_traces: list[dict[str, Any]] | None = None,
    hard_accuracy_threshold: float,
    easy_accuracy_threshold: float,
) -> list[dict[str, Any]]:
    """Build per-question metadata from all traces."""

    raw_by_question = _group_traces_by_question(traces)
    question_ids = sorted(raw_by_question)

    if not 0.0 <= hard_accuracy_threshold <= 1.0:
        raise ValueError("analysis.hard_accuracy_threshold must lie in [0, 1].")
    if not 0.0 <= easy_accuracy_threshold <= 1.0:
        raise ValueError("analysis.easy_accuracy_threshold must lie in [0, 1].")
    if hard_accuracy_threshold >= easy_accuracy_threshold:
        raise ValueError("analysis.hard_accuracy_threshold must be smaller than analysis.easy_accuracy_threshold.")

    metadata_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        raw_rows = raw_by_question.get(question_id, [])
        accuracy = _compute_accuracy(raw_rows)
        difficulty_score = 1.0 - accuracy
        if accuracy > easy_accuracy_threshold:
            difficulty_bucket = "easy"
        elif accuracy < hard_accuracy_threshold:
            difficulty_bucket = "hard"
        else:
            difficulty_bucket = "medium"

        metadata_rows.append(
            {
                "question_id": question_id,
                "difficulty_score": difficulty_score,
                "difficulty_bucket": difficulty_bucket,
                "accuracy": accuracy,
                "total_samples": len(raw_rows),
                "correct_count": sum(1 for trace in raw_rows if trace["is_correct"]),
                "natural_length_distribution": _length_distribution(raw_rows),
            }
        )

    return metadata_rows


def build_accuracy_rows_by_difficulty(
    *,
    traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    min_nldd_length: int | None = None,
) -> list[dict[str, Any]]:
    """Build exact-length accuracy rows for each difficulty bucket."""

    metadata_by_question = {
        str(row["question_id"]): row
        for row in question_metadata
    }
    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTY_ORDER:
        subset = [
            trace
            for trace in traces
            if metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
        grouped: dict[int, list[int]] = defaultdict(list)
        for trace in subset:
            length = int(trace["actual_num_steps"])
            if min_nldd_length is not None and length < min_nldd_length:
                continue
            grouped[length].append(1 if trace["is_correct"] else 0)
        for length in sorted(grouped):
            outcomes = grouped[length]
            mean = sum(outcomes) / len(outcomes)
            rows.append(
                {
                    "difficulty": difficulty,
                    "length": length,
                    "n": len(outcomes),
                    "mean_accuracy": mean,
                    "se_accuracy": _standard_error(outcomes),
                }
            )
    return rows


def _group_traces_by_question(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace["question_id"])].append(trace)
    return grouped


def _compute_accuracy(traces: list[dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    return sum(1 for trace in traces if trace["is_correct"]) / len(traces)


def _standard_error(outcomes: list[int]) -> float:
    if not outcomes:
        return 0.0
    mean = sum(outcomes) / len(outcomes)
    return (mean * (1.0 - mean) / len(outcomes)) ** 0.5


def _length_distribution(traces: list[dict[str, Any]]) -> dict[str, int]:
    distribution = Counter(int(trace["actual_num_steps"]) for trace in traces)
    return {
        str(length): distribution[length]
        for length in sorted(distribution)
    }
