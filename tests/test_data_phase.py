"""Tests for data-phase curation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from src.data_phase2.curation import curate_data_phase
from src.analysis_phase.nldd import summarize_corruption_records
from src.data_phase2.aggregation import aggregate_stage1_outputs


def test_curate_data_phase_restructures_canonical_run(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_data_phase_{uuid.uuid4().hex}"
    canonical = workspace / "full_generation"
    legacy = workspace / "full_run_diff-temperature"
    try:
        _build_valid_run(canonical, dedup=True)
        _build_valid_run(legacy, dedup=False)

        result = curate_data_phase(
            str(canonical),
            legacy_run_dir=str(legacy),
        )

        assert Path(result["manifest_path"]).exists()
        assert Path(result["readme_path"]).exists()
        assert (canonical / "legacy" / "failed_corruptions.jsonl").exists()
        assert (canonical / "legacy" / "stage1_analysis_report.md").exists()
        assert (canonical / "legacy" / "manual_review_items.md").exists()
        assert (canonical / "legacy" / "logs").exists()
        assert (canonical / "legacy" / "shards").exists()
        assert not (canonical / "failed_corruptions.jsonl").exists()

        manifest = json.loads((canonical / "data_phase_manifest.json").read_text(encoding="utf-8"))
        artifacts = {row["path"]: row for row in manifest["artifacts"]}
        assert artifacts["traces.jsonl"]["role"] == "canonical"
        assert artifacts["traces.jsonl"]["default_for_analysis"] is True
        assert artifacts["legacy/failed_corruptions.jsonl"]["role"] == "legacy"
        assert artifacts["full_run_diff-temperature/traces.jsonl"]["role"] == "legacy"
        assert artifacts["full_run_diff-temperature/traces.jsonl"]["default_for_analysis"] is False

        readme = (canonical / "README.md").read_text(encoding="utf-8")
        assert "唯一的正式 data-phase 入口" in readme
        assert "旧口径步骤失败记录" in readme
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_curate_data_phase_rejects_duplicate_trace_ids(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_data_phase_dup_{uuid.uuid4().hex}"
    canonical = workspace / "full_generation"
    legacy = workspace / "full_run_diff-temperature"
    try:
        _build_valid_run(canonical, dedup=True)
        _build_valid_run(legacy, dedup=False)

        with (canonical / "traces.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_trace("q1", "icl_short", 1, True, 2), ensure_ascii=False) + "\n"
            )

        try:
            curate_data_phase(
                str(canonical),
                legacy_run_dir=str(legacy),
            )
        except ValueError as exc:
            assert "duplicate trace_id" in str(exc)
        else:
            raise AssertionError("curate_data_phase should reject duplicate trace ids")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _build_valid_run(run_dir: Path, *, dedup: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = [
        _trace("q1", "icl_short", 1, True, 2),
        _trace("q1", "icl_medium", 1, True, 3),
        _trace("q2", "icl_short", 1, False, 4),
        _trace("q2", "icl_medium", 1, True, 4),
    ]
    if not dedup:
        traces.append(_trace("q2", "icl_medium", 2, True, 4))
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "peak-cot-stage1-gsm8k-platinum-llama31",
                "max_new_tokens": 512,
                "icl_group_sample_counts": {
                    "icl_short": 5,
                    "icl_medium": 5,
                    "icl_detailed": 5,
                    "icl_verbose": 5,
                },
                "schema_version": "stage1_trace_v2",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate_stage1_outputs(str(run_dir))
    _write_corruption_artifacts(run_dir)

    (run_dir / "failed_corruptions.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "q1_icl_short_1",
                "question_id": "q1",
                "step_index": 1,
                "step_text": "Step 1",
                "actual_num_steps": 2,
                "failure_reason": "no_numeric",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stage1_analysis_report.md").write_text("# stale\n", encoding="utf-8")
    (run_dir / "manual_review_items.md").write_text("# stale\n", encoding="utf-8")
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / "dedup.log").write_text("dedup log\n", encoding="utf-8")
    (run_dir / "shards").mkdir(exist_ok=True)
    (run_dir / "shards" / "placeholder.txt").write_text("placeholder\n", encoding="utf-8")


def _write_corruption_artifacts(run_dir: Path) -> None:
    corruption_dir = run_dir / "corruptted_traces"
    corruption_dir.mkdir(parents=True, exist_ok=True)
    all_steps = [
        {
            "corruption_id": "q1_icl_short_1_step1_all_steps",
            "trace_id": "q1_icl_short_1",
            "corruption_failed": False,
            "corruption_tier": 1,
            "corruption_type": "numeric_result",
            "failure_tier": None,
        },
        {
            "corruption_id": "q2_icl_short_1_step1_all_steps",
            "trace_id": "q2_icl_short_1",
            "corruption_failed": True,
            "corruption_tier": 4,
            "corruption_type": "uncorruptible",
            "failure_tier": "uncorruptible",
        },
    ]
    sampled_steps = [
        {
            "corruption_id": "q1_icl_medium_1_step1_sampled_steps",
            "trace_id": "q1_icl_medium_1",
            "corruption_failed": False,
            "corruption_tier": 2,
            "corruption_type": "operator_swap",
            "failure_tier": None,
        }
    ]
    for name, rows in (("all_steps.jsonl", all_steps), ("sampled_steps.jsonl", sampled_steps)):
        with (corruption_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_corruption_records(
        {
            "all_steps": all_steps,
            "sampled_steps": sampled_steps,
        }
    )
    (corruption_dir / "corruption_summary.json").write_text(
        json.dumps({"metadata": {"seed": 42}, "summary": summary}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _trace(
    question_id: str,
    prompt_id: str,
    sample_idx: int,
    is_correct: bool,
    actual_num_steps: int,
) -> dict[str, object]:
    return {
        "trace_id": f"{question_id}_{prompt_id}_{sample_idx}",
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": 1.0,
        "prompt_id": prompt_id,
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
