"""Tests for Stage E accuracy aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

from src.data_phase2.aggregation import (
    AccuracyBucket,
    aggregate_stage1_outputs,
    choose_merge_neighbor,
    discover_stage1_shard_paths,
    merge_sparse_accuracy_buckets,
    merge_stage1_shards,
)


def test_choose_merge_neighbor_prefers_larger_adjacent_bucket() -> None:
    buckets = [
        AccuracyBucket(lengths=[2], outcomes=[1, 0]),
        AccuracyBucket(lengths=[3], outcomes=[1]),
        AccuracyBucket(lengths=[4], outcomes=[1, 1, 1, 0]),
    ]

    assert choose_merge_neighbor(buckets, 1) == 2


def test_merge_sparse_accuracy_buckets_merges_until_threshold() -> None:
    buckets = [
        AccuracyBucket(lengths=[2, 2], outcomes=[1, 1]),
        AccuracyBucket(lengths=[3], outcomes=[1]),
        AccuracyBucket(lengths=[4, 4, 4, 4], outcomes=[0, 0, 1, 1]),
    ]

    merged = merge_sparse_accuracy_buckets(buckets, min_bin_size=3)

    assert len(merged) == 2
    assert merged[0].lengths == [2, 2, 3]
    assert merged[0].n == 3
    assert merged[1].lengths == [4, 4, 4, 4]
    assert merged[1].n == 4


def test_aggregate_stage1_outputs_writes_stage_e_artifacts(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "stage-e-test")

    run_dir = Path("tests") / f"_tmp_stage_e_{uuid.uuid4().hex}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        traces_path = run_dir / "traces.jsonl"

        traces = [
            _trace("q1", 2, True),
            _trace("q1", 2, True),
            _trace("q2", 2, True),
            _trace("q2", 2, False),
            _trace("q2", 2, True),
            _trace("q3", 4, False),
            _trace("q3", 4, True),
            _trace("q4", 5, True),
            _trace("q4", 5, True),
            _trace("q4", 5, True),
            _trace("q4", 5, True),
            _trace("q4", 5, False),
            _trace("q4", 5, True),
            _trace("q4", 8, False),
            _trace("q5", 8, False),
            _trace("q5", 8, True),
            _trace("q6", 8, False),
            _trace("q6", 8, False),
        ]
        with traces_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace) + "\n")

        artifacts = aggregate_stage1_outputs(str(run_dir))

        accuracy_path = Path(artifacts["accuracy_by_length_path"])
        metadata_path = Path(artifacts["question_metadata_path"])
        coarse_analysis_path = Path(artifacts["coarse_analysis_path"])
        lstar_summary_path = Path(artifacts["lstar_summary_path"])
        assert accuracy_path.exists()
        assert metadata_path.exists()
        assert coarse_analysis_path.exists()
        assert lstar_summary_path.exists()

        with accuracy_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert rows
        assert set(rows[0]) == {"difficulty", "dedup_mode", "bucket_label", "n", "mean", "se"}
        assert {row["dedup_mode"] for row in rows} <= {"dedup", "raw"}

        metadata_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata_by_question = {
            row["question_id"]: row
            for row in metadata_rows
        }

        assert metadata_by_question["q1"]["difficulty_bucket"] is None
        assert metadata_by_question["q1"]["excluded_from_difficulty"] is True
        assert metadata_by_question["q1"]["accuracy"] == 1.0
        assert metadata_by_question["q1"]["optimal_length"] == 2
        assert metadata_by_question["q1"]["natural_length_distribution"] == {"2": 2}
        assert metadata_by_question["q4"]["optimal_length"] == 5
        assert metadata_by_question["q5"]["natural_length_distribution"] == {"8": 2}

        coarse = json.loads(coarse_analysis_path.read_text(encoding="utf-8"))
        assert coarse["schema_version"] == "stage1_coarse_analysis_v4"
        assert set(coarse["difficulties"]) == {"easy", "medium", "hard"}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_aggregate_stage1_outputs_materializes_root_traces_from_shards(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "stage-e-shards")

    run_dir = Path("tests") / f"_tmp_stage_e_shards_{uuid.uuid4().hex}"
    try:
        shard_a = run_dir / "shards" / "q0000_0002"
        shard_b = run_dir / "shards" / "q0002_0004"
        shard_a.mkdir(parents=True, exist_ok=True)
        shard_b.mkdir(parents=True, exist_ok=True)

        traces_a = [_trace("q1", 2, True), _trace("q2", 3, False)]
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
    return {
        "trace_id": f"{question_id}_{actual_num_steps}_{int(is_correct)}",
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": 1.0,
        "prompt_id": "icl_short",
        "raw_completion": "stub",
        "steps": ["stub"] * actual_num_steps,
        "actual_num_steps": actual_num_steps,
        "final_answer_line": "#### 1",
        "extracted_answer": 1.0 if is_correct else 0.0,
        "is_correct": is_correct,
        "extraction_failed": False,
        "token_count": 10 + actual_num_steps,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
