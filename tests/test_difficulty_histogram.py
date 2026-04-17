"""Tests for exporting difficulty histograms from question metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.data_phase2.difficulty_histogram import (
    build_difficulty_histogram,
    export_difficulty_histogram,
)


def test_build_difficulty_histogram_uses_closed_final_bin() -> None:
    rows = build_difficulty_histogram([0.0, 0.05, 0.99, 1.0], bin_size=0.05)

    assert len(rows) == 20
    assert rows[0] == {"bin_left": 0.0, "bin_right": 0.05, "count": 1}
    assert rows[1] == {"bin_left": 0.05, "bin_right": 0.1, "count": 1}
    assert rows[-1] == {"bin_left": 0.95, "bin_right": 1.0, "count": 2}
    assert sum(int(row["count"]) for row in rows) == 4


def test_export_difficulty_histogram_writes_expected_csv() -> None:
    temp_dir = Path("tests") / f"_tmp_difficulty_hist_{uuid.uuid4().hex}"
    metadata_path = temp_dir / "question_metadata.jsonl"
    output_path = temp_dir / "difficulty_histogram.csv"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"question_id": "q1", "difficulty_score": 0.0},
            {"question_id": "q2", "difficulty_score": 0.05},
            {"question_id": "q3", "difficulty_score": 1.0},
        ]
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        export_difficulty_histogram(
            question_metadata_path=metadata_path,
            output_path=output_path,
            bin_size=0.05,
        )

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.DictReader(handle))

        assert parsed[0] == {"bin_left": "0.00", "bin_right": "0.05", "count": "1"}
        assert parsed[1] == {"bin_left": "0.05", "bin_right": "0.10", "count": "1"}
        assert parsed[-1] == {"bin_left": "0.95", "bin_right": "1.00", "count": "1"}
        assert sum(int(row["count"]) for row in parsed) == len(rows)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_difficulty_histogram_cli(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    temp_dir = Path("tests") / f"_tmp_difficulty_hist_cli_{uuid.uuid4().hex}"
    metadata_path = temp_dir / "question_metadata.jsonl"
    output_path = temp_dir / "difficulty_histogram.csv"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"question_id": "q1", "difficulty_score": 0.25}) + "\n")
            handle.write(json.dumps({"question_id": "q2", "difficulty_score": 0.75}) + "\n")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/export_difficulty_histogram.py",
                "--question-metadata",
                str(metadata_path),
                "--output",
                str(output_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        assert output_path.exists()
        assert "wrote:" in result.stdout
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
