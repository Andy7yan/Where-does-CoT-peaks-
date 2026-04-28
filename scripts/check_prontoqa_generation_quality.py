"""Quality checks for PrOntoQA Stage 1 generation shards."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.pipeline import discover_stage1_shard_paths


def main() -> None:
    args = parse_args()
    result = check_prontoqa_generation_quality(
        run_dir=args.run_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        shard_id=args.shard_id,
        step_std_threshold=args.step_std_threshold,
        high_accuracy_threshold=args.high_accuracy_threshold,
        low_accuracy_threshold=args.low_accuracy_threshold,
        meaningful_drop_threshold=args.meaningful_drop_threshold,
        min_length_bin_n=args.min_length_bin_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def check_prontoqa_generation_quality(
    *,
    run_dir: str | Path,
    config_path: str | Path = "configs/stage1_prontoqa.yaml",
    output_dir: str | Path | None = None,
    shard_id: str | None = None,
    step_std_threshold: float = 1.5,
    high_accuracy_threshold: float = 0.95,
    low_accuracy_threshold: float = 0.05,
    meaningful_drop_threshold: float = 0.10,
    min_length_bin_n: int = 20,
) -> dict[str, Any]:
    """Write and return the two first-shard quality checks requested for PrOntoQA."""

    run_path = Path(run_dir)
    traces = _load_run_traces(run_path, shard_id=shard_id)
    if not traces:
        raise ValueError(f"No traces found under {run_path}.")

    quality_dir = Path(output_dir) if output_dir is not None else run_path / "quality_checks"
    quality_dir.mkdir(parents=True, exist_ok=True)

    expected_traces_per_question = _expected_traces_per_question(Path(config_path))
    per_question_rows = _build_per_question_step_rows(
        traces=traces,
        expected_traces_per_question=expected_traces_per_question,
    )
    accuracy_rows = _build_accuracy_by_length_rows(traces)
    summary = _build_summary(
        traces=traces,
        per_question_rows=per_question_rows,
        accuracy_rows=accuracy_rows,
        step_std_threshold=step_std_threshold,
        high_accuracy_threshold=high_accuracy_threshold,
        low_accuracy_threshold=low_accuracy_threshold,
        meaningful_drop_threshold=meaningful_drop_threshold,
        min_length_bin_n=min_length_bin_n,
        expected_traces_per_question=expected_traces_per_question,
    )

    per_question_path = quality_dir / "per_question_step_variation.csv"
    accuracy_path = quality_dir / "accuracy_by_length_quality.csv"
    summary_path = quality_dir / "quality_summary.json"
    _write_csv(per_question_path, per_question_rows)
    _write_csv(accuracy_path, accuracy_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "quality_dir": str(quality_dir),
        "per_question_step_variation_path": str(per_question_path),
        "accuracy_by_length_quality_path": str(accuracy_path),
        "quality_summary_path": str(summary_path),
        **summary,
    }


def _load_run_traces(run_path: Path, *, shard_id: str | None) -> list[dict[str, Any]]:
    if shard_id:
        trace_paths = [run_path / "shards" / shard_id / "traces.jsonl"]
    else:
        root_trace_path = run_path / "traces.jsonl"
        trace_paths = [root_trace_path] if root_trace_path.exists() else discover_stage1_shard_paths(run_path)

    rows: list[dict[str, Any]] = []
    for path in trace_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    return rows


def _expected_traces_per_question(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        return None
    default_samples = generation.get("samples_per_group")
    icl_groups = generation.get("icl_groups", {})
    if not isinstance(icl_groups, dict):
        return None
    total = 0
    for group_config in icl_groups.values():
        if isinstance(group_config, dict) and group_config.get("samples_per_group") is not None:
            total += int(group_config["samples_per_group"])
        elif default_samples is not None:
            total += int(default_samples)
    return total if total > 0 else None


def _build_per_question_step_rows(
    *,
    traces: list[dict[str, Any]],
    expected_traces_per_question: int | None,
) -> list[dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        by_question.setdefault(str(trace["question_id"]), []).append(trace)

    rows: list[dict[str, Any]] = []
    for question_id in sorted(by_question):
        question_traces = by_question[question_id]
        lengths = [int(row["actual_num_steps"]) for row in question_traces]
        prompt_ids = sorted({str(row.get("prompt_id", "")) for row in question_traces})
        std_population = _std(lengths, sample=False)
        std_sample = _std(lengths, sample=True)
        rows.append(
            {
                "question_id": question_id,
                "trace_count": len(question_traces),
                "expected_trace_count": expected_traces_per_question,
                "complete": expected_traces_per_question is None
                or len(question_traces) == expected_traces_per_question,
                "step_mean": round(sum(lengths) / len(lengths), 6),
                "step_std": round(std_population, 6),
                "step_std_sample": round(std_sample, 6) if std_sample is not None else "",
                "step_min": min(lengths),
                "step_max": max(lengths),
                "unique_step_counts": len(set(lengths)),
                "prompt_ids": "|".join(prompt_ids),
                "accuracy": round(
                    sum(1 for row in question_traces if bool(row.get("is_correct")))
                    / len(question_traces),
                    6,
                ),
            }
        )
    return rows


def _build_accuracy_by_length_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_length: dict[int, list[dict[str, Any]]] = {}
    for trace in traces:
        by_length.setdefault(int(trace["actual_num_steps"]), []).append(trace)

    rows: list[dict[str, Any]] = []
    for length in sorted(by_length):
        length_traces = by_length[length]
        correct = sum(1 for row in length_traces if bool(row.get("is_correct")))
        rows.append(
            {
                "actual_num_steps": length,
                "n": len(length_traces),
                "correct": correct,
                "accuracy": round(correct / len(length_traces), 6),
            }
        )
    return rows


def _build_summary(
    *,
    traces: list[dict[str, Any]],
    per_question_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    step_std_threshold: float,
    high_accuracy_threshold: float,
    low_accuracy_threshold: float,
    meaningful_drop_threshold: float,
    min_length_bin_n: int,
    expected_traces_per_question: int | None,
) -> dict[str, Any]:
    low_variation_rows = [
        row for row in per_question_rows if float(row["step_std"]) < step_std_threshold
    ]
    incomplete_rows = [row for row in per_question_rows if not bool(row["complete"])]
    low_variation_ratio = len(low_variation_rows) / len(per_question_rows)

    eligible_accuracy_rows = [
        row for row in accuracy_rows if int(row["n"]) >= min_length_bin_n
    ]
    accuracies = [float(row["accuracy"]) for row in eligible_accuracy_rows]
    all_near_one = bool(accuracies) and all(value >= high_accuracy_threshold for value in accuracies)
    all_near_zero = bool(accuracies) and all(value <= low_accuracy_threshold for value in accuracies)
    drops = _adjacent_accuracy_drops(eligible_accuracy_rows)
    max_drop = max((drop["drop"] for drop in drops), default=0.0)
    has_meaningful_drop = max_drop >= meaningful_drop_threshold

    return {
        "num_traces": len(traces),
        "num_questions": len(per_question_rows),
        "expected_traces_per_question": expected_traces_per_question,
        "incomplete_question_count": len(incomplete_rows),
        "step_std_threshold": step_std_threshold,
        "questions_below_step_std_threshold": len(low_variation_rows),
        "questions_below_step_std_threshold_ratio": round(low_variation_ratio, 6),
        "length_variation_status": (
            "needs_strategy_adjustment"
            if low_variation_ratio > 0.5
            else "ok"
        ),
        "min_length_bin_n": min_length_bin_n,
        "eligible_length_bin_count": len(eligible_accuracy_rows),
        "high_accuracy_threshold": high_accuracy_threshold,
        "low_accuracy_threshold": low_accuracy_threshold,
        "meaningful_drop_threshold": meaningful_drop_threshold,
        "all_eligible_lengths_near_one": all_near_one,
        "all_eligible_lengths_near_zero": all_near_zero,
        "max_adjacent_accuracy_drop": round(max_drop, 6),
        "has_meaningful_accuracy_drop": has_meaningful_drop,
        "accuracy_curve_status": (
            "difficulty_distribution_problem"
            if all_near_one or all_near_zero or not has_meaningful_drop
            else "ok"
        ),
        "largest_adjacent_drops": drops[:5],
    }


def _adjacent_accuracy_drops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: int(row["actual_num_steps"]))
    drops: list[dict[str, Any]] = []
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        drop = float(previous["accuracy"]) - float(current["accuracy"])
        drops.append(
            {
                "from_L": int(previous["actual_num_steps"]),
                "to_L": int(current["actual_num_steps"]),
                "from_accuracy": float(previous["accuracy"]),
                "to_accuracy": float(current["accuracy"]),
                "drop": round(drop, 6),
            }
        )
    return sorted(drops, key=lambda row: row["drop"], reverse=True)


def _std(values: list[int], *, sample: bool) -> float | None:
    if not values:
        return None
    if sample and len(values) < 2:
        return None
    mean = sum(values) / len(values)
    denominator = len(values) - 1 if sample else len(values)
    variance = sum((value - mean) ** 2 for value in values) / denominator
    return sqrt(variance)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory with traces or shard traces.")
    parser.add_argument(
        "--config",
        default="configs/stage1_prontoqa.yaml",
        help="Config used to infer expected traces per question.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-dir>/quality_checks.")
    parser.add_argument("--shard-id", default=None, help="Check only one shard, e.g. q0000_0250.")
    parser.add_argument("--step-std-threshold", type=float, default=1.5)
    parser.add_argument("--high-accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--low-accuracy-threshold", type=float, default=0.05)
    parser.add_argument("--meaningful-drop-threshold", type=float, default=0.10)
    parser.add_argument("--min-length-bin-n", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
