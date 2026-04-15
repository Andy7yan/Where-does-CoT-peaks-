"""Helpers for Stage 1 v4 coarse analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any


DIFFICULTY_ORDER = ("easy", "medium", "hard")
LENGTH_BIN_ORDER = ("short", "mid", "long")


@dataclass(frozen=True)
class DifficultyThresholds:
    """Resolved accuracy thresholds for difficulty binning."""

    hard: float
    easy: float


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


def build_question_metadata_v4(
    *,
    traces: list[dict[str, Any]],
    deduped_traces: list[dict[str, Any]],
    difficulty_quantiles: list[float],
    accuracy_exclusion_bounds: list[float],
) -> list[dict[str, Any]]:
    """Build per-question metadata with v4 difficulty annotations."""

    raw_by_question = _group_traces_by_question(traces)
    deduped_by_question = _group_traces_by_question(deduped_traces)
    question_ids = sorted(set(raw_by_question) | set(deduped_by_question))

    if len(difficulty_quantiles) != 2:
        raise ValueError("analysis.difficulty_quantiles must contain exactly two entries.")
    if len(accuracy_exclusion_bounds) != 2:
        raise ValueError("analysis.accuracy_exclusion_bounds must contain exactly two entries.")

    lower_exclusion, upper_exclusion = accuracy_exclusion_bounds
    per_question_dedup_accuracy = {
        question_id: _compute_accuracy(deduped_by_question.get(question_id, []))
        for question_id in question_ids
    }
    threshold_inputs = [
        accuracy
        for accuracy in per_question_dedup_accuracy.values()
        if lower_exclusion <= accuracy <= upper_exclusion
    ]
    if not threshold_inputs:
        threshold_inputs = list(per_question_dedup_accuracy.values())
    thresholds = DifficultyThresholds(
        hard=interpolated_quantile(threshold_inputs, difficulty_quantiles[0]),
        easy=interpolated_quantile(threshold_inputs, difficulty_quantiles[1]),
    )

    metadata_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        raw_rows = raw_by_question.get(question_id, [])
        dedup_rows = deduped_by_question.get(question_id, [])
        raw_accuracy = _compute_accuracy(raw_rows)
        dedup_accuracy = per_question_dedup_accuracy[question_id]
        excluded = dedup_accuracy < lower_exclusion or dedup_accuracy > upper_exclusion
        difficulty_bucket = None
        if not excluded:
            if dedup_accuracy > thresholds.easy:
                difficulty_bucket = "easy"
            elif dedup_accuracy < thresholds.hard:
                difficulty_bucket = "hard"
            else:
                difficulty_bucket = "medium"

        metadata_rows.append(
            {
                "question_id": question_id,
                "difficulty": 1.0 - dedup_accuracy,
                "difficulty_score": 1.0 - dedup_accuracy,
                "difficulty_bucket": difficulty_bucket,
                "excluded_from_difficulty": excluded,
                "difficulty_threshold_hard": thresholds.hard,
                "difficulty_threshold_easy": thresholds.easy,
                "accuracy": dedup_accuracy,
                "accuracy_raw": raw_accuracy,
                "accuracy_dedup": dedup_accuracy,
                "optimal_length": _compute_optimal_length(dedup_rows),
                "optimal_length_raw": _compute_optimal_length(raw_rows),
                "total_samples": len(raw_rows),
                "correct_count": sum(1 for trace in raw_rows if trace["is_correct"]),
                "dedup_total_samples": len(dedup_rows),
                "dedup_correct_count": sum(1 for trace in dedup_rows if trace["is_correct"]),
                "natural_length_distribution": _length_distribution(raw_rows),
                "dedup_natural_length_distribution": _length_distribution(dedup_rows),
            }
        )

    return metadata_rows


def build_accuracy_buckets_by_difficulty(
    *,
    traces: list[dict[str, Any]],
    deduped_traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    build_accuracy_buckets: Any,
    min_bin_size: int,
) -> dict[str, dict[str, list[Any]]]:
    """Build raw and deduped accuracy buckets for each difficulty bin."""

    metadata_by_question = {
        str(row["question_id"]): row
        for row in question_metadata
    }
    rows: dict[str, dict[str, list[Any]]] = {
        difficulty: {"raw": [], "dedup": []}
        for difficulty in DIFFICULTY_ORDER
    }
    for difficulty in DIFFICULTY_ORDER:
        raw_subset = [
            trace
            for trace in traces
            if metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
        dedup_subset = [
            trace
            for trace in deduped_traces
            if metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
        if raw_subset:
            rows[difficulty]["raw"] = build_accuracy_buckets(raw_subset, min_bin_size=min_bin_size)
        if dedup_subset:
            rows[difficulty]["dedup"] = build_accuracy_buckets(dedup_subset, min_bin_size=min_bin_size)
    return rows


def build_coarse_analysis(
    *,
    deduped_traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    buckets_by_difficulty: dict[str, dict[str, list[Any]]],
    min_nldd_length: int,
    primary_lstar_window: int,
    fallback_lstar_window: int,
    min_near_lstar_traces: int,
    min_cell_size: int,
    select_l_star: Any,
) -> dict[str, Any]:
    """Build the frozen coarse-analysis summary required by v4."""

    metadata_by_question = {
        str(row["question_id"]): row
        for row in question_metadata
    }
    threshold_row = next(iter(question_metadata), None)
    thresholds = {
        "hard": threshold_row["difficulty_threshold_hard"] if threshold_row else None,
        "easy": threshold_row["difficulty_threshold_easy"] if threshold_row else None,
    }

    difficulties: dict[str, Any] = {}
    for difficulty in DIFFICULTY_ORDER:
        difficulty_rows = [
            trace
            for trace in deduped_traces
            if metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
        correct_rows = [
            trace
            for trace in difficulty_rows
            if trace["is_correct"] and int(trace["actual_num_steps"]) >= min_nldd_length
        ]
        l_star = select_l_star(buckets_by_difficulty[difficulty]["dedup"])
        q33 = None
        q67 = None
        initial_counts = {label: 0 for label in LENGTH_BIN_ORDER}
        merged_bins: list[dict[str, Any]] = []
        length_bin_map = {label: None for label in LENGTH_BIN_ORDER}
        if correct_rows:
            lengths = [int(trace["actual_num_steps"]) for trace in correct_rows]
            q33 = interpolated_quantile(lengths, 1.0 / 3.0)
            q67 = interpolated_quantile(lengths, 2.0 / 3.0)
            for trace in correct_rows:
                initial_counts[assign_length_bin(int(trace["actual_num_steps"]), q33, q67)] += 1
            merged_bins = merge_length_bin_counts(initial_counts, min_cell_size=min_cell_size)
            for entry in merged_bins:
                for member in entry["members"]:
                    length_bin_map[member] = entry["label"]

        near_lstar = build_near_lstar_summary(
            correct_rows=correct_rows,
            l_star=l_star,
            primary_radius=primary_lstar_window,
            fallback_radius=fallback_lstar_window,
            min_required=min_near_lstar_traces,
        )
        difficulties[difficulty] = {
            "question_count": sum(
                1
                for row in question_metadata
                if row["difficulty_bucket"] == difficulty
            ),
            "trace_count": len(difficulty_rows),
            "correct_trace_count_min_nldd_length": len(correct_rows),
            "l_star": l_star,
            "length_tertiles": {
                "q33": q33,
                "q67": q67,
            },
            "initial_length_bin_counts": initial_counts,
            "merged_length_bins": merged_bins,
            "length_bin_map": length_bin_map,
            "near_lstar": near_lstar,
        }

    return {
        "schema_version": "stage1_coarse_analysis_v4",
        "difficulty_thresholds": thresholds,
        "difficulties": difficulties,
        "notes": {
            "difficulty_source": "deduplicated per-question accuracy",
            "length_bin_mode": "tertile",
            "min_nldd_length": min_nldd_length,
            "primary_lstar_window": primary_lstar_window,
            "fallback_lstar_window": fallback_lstar_window,
            "min_near_lstar_traces": min_near_lstar_traces,
            "min_cell_size": min_cell_size,
        },
    }


def build_near_lstar_summary(
    *,
    correct_rows: list[dict[str, Any]],
    l_star: float | None,
    primary_radius: int,
    fallback_radius: int,
    min_required: int,
) -> dict[str, Any]:
    """Resolve the frozen near-L* window for one difficulty bucket."""

    if l_star is None:
        return {
            "primary_window": None,
            "primary_count": 0,
            "fallback_window": None,
            "fallback_count": 0,
            "selected_window": None,
            "selected_count": 0,
            "status": "missing_lstar",
        }

    primary_window = [l_star - primary_radius, l_star + primary_radius]
    primary_count = count_lengths_in_window(correct_rows, primary_window)
    fallback_window = [l_star - fallback_radius, l_star + fallback_radius]
    fallback_count = count_lengths_in_window(correct_rows, fallback_window)

    if primary_count >= min_required:
        selected_window = primary_window
        selected_count = primary_count
        status = "primary"
    elif fallback_count >= min_required:
        selected_window = fallback_window
        selected_count = fallback_count
        status = "fallback"
    elif correct_rows:
        selected_window = fallback_window
        selected_count = fallback_count
        status = "insufficient_after_fallback"
    else:
        selected_window = None
        selected_count = 0
        status = "missing_correct_traces"

    return {
        "primary_window": primary_window,
        "primary_count": primary_count,
        "fallback_window": fallback_window,
        "fallback_count": fallback_count,
        "selected_window": selected_window,
        "selected_count": selected_count,
        "status": status,
    }


def count_lengths_in_window(rows: list[dict[str, Any]], window: list[float]) -> int:
    """Count traces whose lengths fall inside an inclusive window."""

    left, right = window
    return sum(
        1
        for row in rows
        if left <= int(row["actual_num_steps"]) <= right
    )


def merge_length_bin_counts(
    counts: dict[str, int],
    *,
    min_cell_size: int,
) -> list[dict[str, Any]]:
    """Merge sparse length bins according to the v4 adjacency rule."""

    active = [
        {"members": [label], "count": counts[label]}
        for label in LENGTH_BIN_ORDER
        if counts[label] > 0
    ]
    if not active:
        return []

    while len(active) > 1:
        sparse_index = next(
            (index for index, entry in enumerate(active) if entry["count"] < min_cell_size),
            None,
        )
        if sparse_index is None:
            break
        neighbor_index = choose_length_merge_neighbor(active, sparse_index)
        left_index, right_index = sorted((sparse_index, neighbor_index))
        merged = {
            "members": active[left_index]["members"] + active[right_index]["members"],
            "count": active[left_index]["count"] + active[right_index]["count"],
        }
        active[left_index:right_index + 1] = [merged]

    result: list[dict[str, Any]] = []
    nonempty_count = len([label for label in LENGTH_BIN_ORDER if counts[label] > 0])
    for entry in active:
        result.append(
            {
                "label": "_".join(entry["members"]),
                "members": list(entry["members"]),
                "count": entry["count"],
                "merged_all": len(entry["members"]) == nonempty_count,
            }
        )
    return result


def choose_length_merge_neighbor(
    active: list[dict[str, Any]],
    sparse_index: int,
) -> int:
    """Choose the adjacent length bin used to absorb a sparse cell."""

    if sparse_index == 0:
        return 1
    if sparse_index == len(active) - 1:
        return len(active) - 2
    if active[sparse_index - 1]["count"] <= active[sparse_index + 1]["count"]:
        return sparse_index - 1
    return sparse_index + 1


def assign_length_bin(length: int, q33: float, q67: float) -> str:
    """Assign one trace length into the per-difficulty tertile bins."""

    if length <= q33:
        return "short"
    if length <= q67:
        return "mid"
    return "long"


def interpolated_quantile(values: list[int] | list[float], quantile: float) -> float:
    """Compute a linear-interpolated quantile."""

    if not values:
        raise ValueError("Cannot compute a quantile from an empty collection.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1].")

    sorted_values = sorted(float(value) for value in values)
    index = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (index - lower)


def _group_traces_by_question(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace["question_id"])].append(trace)
    return grouped


def _compute_accuracy(traces: list[dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    return sum(1 for trace in traces if trace["is_correct"]) / len(traces)


def _compute_optimal_length(traces: list[dict[str, Any]]) -> int | None:
    grouped_accuracy: dict[int, list[int]] = defaultdict(list)
    for trace in traces:
        grouped_accuracy[int(trace["actual_num_steps"])].append(1 if trace["is_correct"] else 0)
    if not grouped_accuracy:
        return None
    return min(
        grouped_accuracy,
        key=lambda length: (
            -sum(grouped_accuracy[length]) / len(grouped_accuracy[length]),
            length,
        ),
    )


def _length_distribution(traces: list[dict[str, Any]]) -> dict[str, int]:
    distribution = Counter(int(trace["actual_num_steps"]) for trace in traces)
    return {
        str(length): distribution[length]
        for length in sorted(distribution)
    }
