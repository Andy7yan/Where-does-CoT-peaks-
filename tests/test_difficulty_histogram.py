"""Tests for exporting difficulty histograms from question metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.data_phase2.difficulty_profile import export_difficulty_profile
from src.data_phase2.difficulty_histogram import (
    build_difficulty_histogram,
    export_difficulty_histogram,
)
from scripts.check_prontoqa_generation_quality import check_prontoqa_generation_quality
from scripts.check_prontoqa_env_consistency import build_report


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


def test_export_difficulty_profile_writes_prebucket_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    temp_dir = Path("tests") / f"_tmp_difficulty_profile_{uuid.uuid4().hex}"
    run_dir = temp_dir / "run"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        traces = [
            _trace("q1", True, 2),
            _trace("q1", True, 3),
            _trace("q2", False, 4),
            _trace("q2", True, 4),
        ]
        with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
            for row in traces:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        result = export_difficulty_profile(
            run_dir=run_dir,
            config_path="configs/stage1_prontoqa.yaml",
            write_plot=False,
        )

        profile_dir = run_dir / "difficulty_profile"
        assert Path(result["question_metadata_path"]).exists()
        assert (profile_dir / "difficulty_histogram.csv").exists()
        assert (profile_dir / "difficulty_summary.json").exists()
        assert result["num_questions"] == 2
        assert result["num_traces"] == 4
        assert result["current_config"]["hard_if_difficulty_score_gt"] == 0.5
        assert abs(result["current_config"]["easy_if_difficulty_score_lt"] - 0.2) < 1e-9
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_check_prontoqa_generation_quality_flags_low_step_variation(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    temp_dir = Path("tests") / f"_tmp_prontoqa_quality_{uuid.uuid4().hex}"
    run_dir = temp_dir / "run"
    shard_dir = run_dir / "shards" / "q0000_0002"
    try:
        shard_dir.mkdir(parents=True, exist_ok=True)
        traces = [
            *[_trace("q1", True, 4) for _ in range(20)],
            *[_trace("q2", True, length) for length in range(2, 22)],
        ]
        with (shard_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(traces):
                row = dict(row)
                row["trace_id"] = f"{row['question_id']}_icl_short_{index}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        result = check_prontoqa_generation_quality(
            run_dir=run_dir,
            config_path="configs/stage1_prontoqa.yaml",
            shard_id="q0000_0002",
            min_length_bin_n=1,
        )

        assert Path(result["per_question_step_variation_path"]).exists()
        assert Path(result["accuracy_by_length_quality_path"]).exists()
        assert result["num_questions"] == 2
        assert result["expected_traces_per_question"] == 20
        assert result["questions_below_step_std_threshold"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_check_prontoqa_env_consistency_builds_ok_report(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hf-home/hub")
    monkeypatch.setenv("HF_DATASETS_CACHE", "/tmp/hf-home/datasets")

    report = build_report(config_path="configs/stage1_prontoqa.yaml", check_torch=False)

    assert report["config"]["task_name"] == "prontoqa_paper"
    assert report["dataset_check"]["question_count"] == 1000
    assert report["prompts"]["prompt_ids"] == [
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ]
    assert report["answer_extraction_check"]["extracted"] == "A"
    assert report["issues"] == []


def _trace(question_id: str, is_correct: bool, actual_num_steps: int) -> dict[str, object]:
    return {
        "trace_id": f"{question_id}_icl_short_{actual_num_steps}",
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": "A",
        "prompt_id": "icl_short",
        "raw_completion": "Final Answer: A",
        "steps": ["Step 1"],
        "actual_num_steps": actual_num_steps,
        "final_answer_line": "Final Answer: A",
        "extracted_answer": "A",
        "is_correct": is_correct,
        "extraction_failed": False,
        "token_count": 10,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
