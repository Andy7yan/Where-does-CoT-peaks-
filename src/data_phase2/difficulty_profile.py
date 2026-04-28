"""Pre-bucketing difficulty profiling for completed Stage 1 generation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.common.settings import ExperimentConfig
from src.data_phase2.coarse_analysis import build_question_difficulty_profile
from src.data_phase2.difficulty_histogram import export_difficulty_histogram
from src.data_phase2.pipeline import load_stage1_traces


def export_difficulty_profile(
    *,
    run_dir: str | Path,
    config_path: str = "configs/stage1.yaml",
    output_dir: str | Path | None = None,
    bin_size: float = 0.05,
    write_plot: bool = True,
) -> dict[str, Any]:
    """Write difficulty scores, histogram, and threshold guidance before final bucketing."""

    run_path = Path(run_dir)
    profile_dir = Path(output_dir) if output_dir is not None else run_path / "difficulty_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.from_yaml(config_path)
    traces = load_stage1_traces(run_path)
    rows = build_question_difficulty_profile(traces=traces)

    metadata_path = profile_dir / "question_metadata_unbucketed.jsonl"
    histogram_path = profile_dir / "difficulty_histogram.csv"
    summary_path = profile_dir / "difficulty_summary.json"
    plot_path = profile_dir / "difficulty_histogram.png"

    _write_jsonl(metadata_path, rows)
    export_difficulty_histogram(
        question_metadata_path=metadata_path,
        output_path=histogram_path,
        bin_size=bin_size,
    )
    if write_plot:
        _try_write_histogram_plot(histogram_path, plot_path)

    summary = build_difficulty_profile_summary(
        rows=rows,
        traces_count=len(traces),
        hard_accuracy_threshold=config.analysis.hard_accuracy_threshold,
        easy_accuracy_threshold=config.analysis.easy_accuracy_threshold,
    )
    _write_json(summary_path, summary)

    return {
        "profile_dir": str(profile_dir),
        "question_metadata_path": str(metadata_path),
        "difficulty_histogram_path": str(histogram_path),
        "difficulty_histogram_plot_path": str(plot_path) if plot_path.exists() else None,
        "difficulty_summary_path": str(summary_path),
        **summary,
    }


def build_difficulty_profile_summary(
    *,
    rows: list[dict[str, Any]],
    traces_count: int,
    hard_accuracy_threshold: float | None,
    easy_accuracy_threshold: float | None,
) -> dict[str, Any]:
    """Summarize a pre-bucketed difficulty profile for boundary selection."""

    scores = sorted(float(row["difficulty_score"]) for row in rows)
    accuracies = sorted(float(row["accuracy"]) for row in rows)
    summary: dict[str, Any] = {
        "num_questions": len(rows),
        "num_traces": traces_count,
        "difficulty_score_quantiles": _quantiles(scores),
        "accuracy_quantiles": _quantiles(accuracies),
        "current_config": {
            "hard_accuracy_threshold": hard_accuracy_threshold,
            "easy_accuracy_threshold": easy_accuracy_threshold,
            "hard_if_difficulty_score_gt": (
                None if hard_accuracy_threshold is None else 1.0 - hard_accuracy_threshold
            ),
            "easy_if_difficulty_score_lt": (
                None if easy_accuracy_threshold is None else 1.0 - easy_accuracy_threshold
            ),
        },
    }
    return summary


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "p25", "p33", "p50", "p67", "p75", "p90", "max")}
    return {
        "min": values[0],
        "p10": _linear_quantile(values, 0.10),
        "p25": _linear_quantile(values, 0.25),
        "p33": _linear_quantile(values, 1.0 / 3.0),
        "p50": _linear_quantile(values, 0.50),
        "p67": _linear_quantile(values, 2.0 / 3.0),
        "p75": _linear_quantile(values, 0.75),
        "p90": _linear_quantile(values, 0.90),
        "max": values[-1],
    }


def _linear_quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _try_write_histogram_plot(histogram_path: Path, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: could not render difficulty histogram plot: {type(exc).__name__}: {exc}")
        return

    rows: list[dict[str, str]] = []
    with histogram_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    left_edges = [float(row["bin_left"]) for row in rows]
    widths = [float(row["bin_right"]) - float(row["bin_left"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(left_edges, counts, width=widths, align="edge", color="#4c78a8", edgecolor="white")
    ax.set_xlabel("difficulty_score = 1 - question_accuracy")
    ax.set_ylabel("question count")
    ax.set_title("Question Difficulty Histogram")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


__all__ = [
    "build_difficulty_profile_summary",
    "export_difficulty_profile",
]
