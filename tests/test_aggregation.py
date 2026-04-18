"""Tests for exact-length data-phase aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

from src.data_phase2.aggregation import (
    aggregate_stage1_outputs,
    build_accuracy_rows,
    discover_stage1_shard_paths,
    merge_stage1_shards,
    select_l_star_from_accuracy_rows,
)


def test_build_accuracy_rows_and_select_l_star_follow_exact_lengths() -> None:
    traces = [
        _trace("q1", 3, True),
        _trace("q1", 3, False),
        _trace("q2", 4, True),
        _trace("q2", 4, True),
        _trace("q3", 5, False),
    ]
    metadata_by_question = {
        "q1": {"difficulty_bucket": "easy"},
        "q2": {"difficulty_bucket": "easy"},
        "q3": {"difficulty_bucket": "easy"},
    }

    rows = build_accuracy_rows(
        traces,
        metadata_by_question=metadata_by_question,
        difficulty="easy",
        min_nldd_length=3,
    )

    assert rows == [
        {"difficulty": "easy", "length": 3, "n": 2, "mean_accuracy": 0.5, "se_accuracy": rows[0]["se_accuracy"]},
        {"difficulty": "easy", "length": 4, "n": 2, "mean_accuracy": 1.0, "se_accuracy": rows[1]["se_accuracy"]},
        {"difficulty": "easy", "length": 5, "n": 1, "mean_accuracy": 0.0, "se_accuracy": rows[2]["se_accuracy"]},
    ]
    assert select_l_star_from_accuracy_rows(rows) == 4


def test_aggregate_stage1_outputs_writes_current_data_phase_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    run_dir = Path("tests") / f"_tmp_stage_e_{uuid.uuid4().hex}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        traces_path = run_dir / "traces.jsonl"

        traces = []
        for index in range(31):
            traces.append(_trace(f"easy_{index:03d}", 4, True))
        for index in range(130):
            traces.append(_trace(f"medium_{index:03d}", 5, True))
        for index in range(2):
            traces.append(_trace(f"too_short_{index:03d}", 2, True))
            traces.append(_trace(f"small_bin_{index:03d}", 6, True))

        with traces_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace) + "\n")

        artifacts = aggregate_stage1_outputs(str(run_dir))

        accuracy_path = Path(artifacts["accuracy_by_length_path"])
        metadata_path = Path(artifacts["question_metadata_path"])
        histogram_path = Path(artifacts["difficulty_histogram_path"])
        assert accuracy_path.exists()
        assert metadata_path.exists()
        assert histogram_path.exists()
        assert "coarse_analysis_path" not in artifacts
        assert "lstar_summary_path" not in artifacts

        with accuracy_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert set(rows[0]) == {"difficulty", "length", "n", "mean_accuracy", "se_accuracy"}

        metadata_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata_by_question = {row["question_id"]: row for row in metadata_rows}
        first_easy = metadata_by_question["easy_000"]
        assert set(first_easy) == {
            "question_id",
            "difficulty_score",
            "difficulty_bucket",
            "accuracy",
            "total_samples",
            "correct_count",
            "natural_length_distribution",
        }
        assert first_easy["difficulty_bucket"] == "easy"

        easy_dir = run_dir / "difficulty" / "easy"
        assert (easy_dir / "questions.jsonl").exists()
        assert (easy_dir / "traces.jsonl").exists()
        assert (easy_dir / "question_metadata.jsonl").exists()
        assert (easy_dir / "bins" / "bin_4" / "selection.jsonl").exists()
        assert not (easy_dir / "bins" / "bin_2").exists()
        assert not (easy_dir / "bins" / "bin_6").exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_aggregate_stage1_outputs_materializes_root_traces_from_shards(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    run_dir = Path("tests") / f"_tmp_stage_e_shards_{uuid.uuid4().hex}"
    try:
        shard_a = run_dir / "shards" / "q0000_0002"
        shard_b = run_dir / "shards" / "q0002_0004"
        shard_a.mkdir(parents=True, exist_ok=True)
        shard_b.mkdir(parents=True, exist_ok=True)

        traces_a = [_trace("q1", 3, True), _trace("q2", 4, False)]
        traces_b = [_trace("q3", 4, True), _trace("q4", 5, True)]
        for shard_dir, rows in ((shard_a, traces_a), (shard_b, traces_b)):
            with (shard_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            (shard_dir / "run_meta.json").write_text(
                json.dumps({"schema_version": "stage1_trace_v2"}, indent=2) + "\n",
                encoding="utf-8",
            )

        shard_paths = discover_stage1_shard_paths(run_dir)
        assert [path.parent.name for path in shard_paths] == ["q0000_0002", "q0002_0004"]

        merged = merge_stage1_shards(shard_paths)
        assert [row["question_id"] for row in merged] == ["q1", "q2", "q3", "q4"]

        artifacts = aggregate_stage1_outputs(str(run_dir))

        assert Path(artifacts["accuracy_by_length_path"]).exists()
        assert (run_dir / "traces.jsonl").exists()
        assert (run_dir / "run_meta.json").exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _trace(question_id: str, actual_num_steps: int, is_correct: bool) -> dict[str, object]:
    steps = [
        f"Step {index}: {index + 2} + 1 = {index + 3}."
        for index in range(actual_num_steps)
    ]
    return {
        "trace_id": f"{question_id}_{actual_num_steps}_{int(is_correct)}",
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": 1.0,
        "prompt_id": "icl_short",
        "raw_completion": "\n".join(steps),
        "steps": steps,
        "actual_num_steps": actual_num_steps,
        "final_answer_line": "#### 1",
        "extracted_answer": 1.0 if is_correct else 0.0,
        "is_correct": is_correct,
        "extraction_failed": False,
        "token_count": 10 + actual_num_steps,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
