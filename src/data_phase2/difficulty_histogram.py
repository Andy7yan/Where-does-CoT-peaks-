"""Helpers for exporting per-question difficulty histograms from metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_question_difficulties(question_metadata_path: str | Path) -> list[float]:
    """Load per-question difficulty scores from a question_metadata JSONL file."""

    path = Path(question_metadata_path)
    difficulties: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            difficulty = row.get("difficulty_score", row.get("difficulty"))
            if not isinstance(difficulty, (int, float)):
                raise TypeError(
                    "question_metadata.jsonl rows must contain numeric 'difficulty_score' "
                    "(or legacy 'difficulty')."
                )
            score = float(difficulty)
            if not (0.0 <= score <= 1.0):
                raise ValueError("difficulty scores must be within [0.0, 1.0].")
            difficulties.append(score)
    return difficulties


def build_difficulty_histogram(
    difficulties: list[float],
    *,
    bin_size: float = 0.05,
) -> list[dict[str, float | int]]:
    """Bucket difficulty scores into a closed histogram over [0, 1]."""

    if bin_size <= 0.0 or bin_size > 1.0:
        raise ValueError("bin_size must be within (0.0, 1.0].")

    num_bins = round(1.0 / bin_size)
    if abs((num_bins * bin_size) - 1.0) > 1e-9:
        raise ValueError("bin_size must divide the [0, 1] interval exactly.")

    counts = [0 for _ in range(num_bins)]
    for difficulty in difficulties:
        if not (0.0 <= difficulty <= 1.0):
            raise ValueError("difficulty scores must be within [0.0, 1.0].")
        index = min(int(difficulty / bin_size), num_bins - 1)
        counts[index] += 1

    rows: list[dict[str, float | int]] = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "bin_left": round(index * bin_size, 10),
                "bin_right": round((index + 1) * bin_size, 10),
                "count": count,
            }
        )
    return rows


def write_difficulty_histogram_csv(
    output_path: str | Path,
    rows: list[dict[str, float | int]],
) -> None:
    """Write histogram rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "bin_left": f"{float(row['bin_left']):.2f}",
                    "bin_right": f"{float(row['bin_right']):.2f}",
                    "count": int(row["count"]),
                }
            )


def export_difficulty_histogram(
    *,
    question_metadata_path: str | Path,
    output_path: str | Path,
    bin_size: float = 0.05,
) -> str:
    """Build and write a difficulty histogram CSV from question metadata."""

    rows = build_difficulty_histogram(
        load_question_difficulties(question_metadata_path),
        bin_size=bin_size,
    )
    write_difficulty_histogram_csv(output_path, rows)
    return str(Path(output_path))
