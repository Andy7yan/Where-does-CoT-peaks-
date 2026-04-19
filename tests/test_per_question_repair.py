"""Tests for per-question repair planning."""

import json
from pathlib import Path
import shutil
import uuid

from src.data_phase1.per_question_repair import (
    REPAIR_MANIFEST_FILENAME,
    REPAIR_REPORT_FILENAME,
    build_repair_bundle,
)
from src.data_phase1.per_question_selection import save_per_question_manifest


def test_build_repair_bundle_separates_append_safe_and_unsafe_issues() -> None:
    temp_dir = Path("tests") / f"_tmp_per_question_repair_{uuid.uuid4().hex}"
    try:
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "question_id": "gsm8k_platinum_0000",
                "question_text": "What is 2 + 2?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0001",
                "question_text": "What is 3 + 1?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0002",
                "question_text": "What is 5 x 5?",
                "gold_answer": 25.0,
                "source_difficulty_bucket": "hard",
                "target_total_traces": 300,
                "target_samples_per_prompt": 75,
            },
        ]
        selection_metadata = {
            "schema_version": "per_question_selection_v1",
            "pipeline_variant": "per_question",
            "source_run_dir": "/srv/scratch/test/peak-CoT/runs/source-run",
            "source_run_name": "source-run",
            "source_question_metadata_path": "/srv/scratch/test/peak-CoT/runs/source-run/question_metadata.jsonl",
            "selected_question_count": 3,
            "per_question_trace_policy": {
                "medium": {"target_total_traces": 120, "target_samples_per_prompt": 30},
                "hard": {"target_total_traces": 300, "target_samples_per_prompt": 75},
            },
        }
        save_per_question_manifest(run_dir, manifest=manifest, selection_metadata=selection_metadata)

        shard_dir = run_dir / "shards" / "q0000_0002"
        shard_dir.mkdir(parents=True, exist_ok=True)
        traces_path = shard_dir / "traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as handle:
            for index in range(120):
                handle.write(
                    json.dumps(
                        {
                            "trace_id": f"q0-{index}",
                            "question_id": "gsm8k_platinum_0000",
                        }
                    )
                    + "\n"
                )
            for index in range(12):
                handle.write(
                    json.dumps(
                        {
                            "trace_id": f"q1-{index}",
                            "question_id": "gsm8k_platinum_0001",
                        }
                    )
                    + "\n"
                )

        report = build_repair_bundle(run_dir, questions_per_shard=2)

        assert report["issue_count"] == 2
        assert report["append_safe_issue_count"] == 1
        assert report["append_unsafe_issue_count"] == 1
        assert report["repair_manifest_count"] == 1

        repair_manifest_path = run_dir / "repair" / REPAIR_MANIFEST_FILENAME
        assert repair_manifest_path.exists()
        repair_rows = [
            json.loads(line)
            for line in repair_manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["question_id"] for row in repair_rows] == ["gsm8k_platinum_0002"]

        report_path = run_dir / "repair" / REPAIR_REPORT_FILENAME
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert report_payload["shard_status"][0]["questions_mismatched"] == 1
        assert report_payload["shard_status"][1]["questions_missing"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_repair_bundle_can_exclude_an_active_shard_from_repair_manifest() -> None:
    temp_dir = Path("tests") / f"_tmp_per_question_repair_exclude_{uuid.uuid4().hex}"
    try:
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "question_id": "gsm8k_platinum_0000",
                "question_text": "What is 2 + 2?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0001",
                "question_text": "What is 3 + 1?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
        ]
        selection_metadata = {
            "schema_version": "per_question_selection_v1",
            "pipeline_variant": "per_question",
            "selected_question_count": 2,
        }
        save_per_question_manifest(run_dir, manifest=manifest, selection_metadata=selection_metadata)

        report = build_repair_bundle(
            run_dir,
            questions_per_shard=1,
            exclude_shard_ids={"q0001_0002"},
        )

        assert report["issue_count"] == 2
        assert report["excluded_issue_count"] == 1
        assert report["repair_manifest_count"] == 1

        repair_rows = [
            json.loads(line)
            for line in (run_dir / "repair" / REPAIR_MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["question_id"] for row in repair_rows] == ["gsm8k_platinum_0000"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
