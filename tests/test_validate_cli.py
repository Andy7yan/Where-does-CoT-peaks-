"""Tests for the validation CLI."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.analysis_phase.nldd import summarize_corruption_records
from src.data_phase2.aggregation import aggregate_stage1_outputs


def test_validate_cli_emits_json(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_validate_cli_{uuid.uuid4().hex}"
    run_dir = workspace / "full_generation"
    try:
        _build_valid_run(run_dir)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate.py",
                "--canonical-run-dir",
                str(run_dir),
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout)
        assert payload["trace_count"] == 4
        assert payload["question_count"] == 2
        assert payload["accuracy_csv_matches_traces"] is True
        assert payload["corruption_summary_matches_records"] is True
        assert payload["difficulty_exports_match_traces"] is True
        assert payload["corruption_validation"]["all_steps"]["records"] == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_validate_cli_accepts_flat_corruption_layout(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_validate_cli_{uuid.uuid4().hex}"
    run_dir = workspace / "full_generation"
    try:
        _build_valid_run(run_dir)
        for name in ("all_steps.jsonl", "corruption_summary.json"):
            payload = (run_dir / "corruptted_traces" / name).read_text(encoding="utf-8")
            (run_dir / name).write_text(payload, encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate.py",
                "--canonical-run-dir",
                str(run_dir),
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout)
        assert payload["corruption_validation"]["all_steps"]["records"] == 2
        assert payload["difficulty_exports_match_traces"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _build_valid_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = [
        _trace("q1", "icl_short", 1, True, 2),
        _trace("q1", "icl_medium", 1, True, 3),
        _trace("q2", "icl_short", 1, False, 4),
        _trace("q2", "icl_medium", 1, True, 4),
    ]
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
    for name, rows in (("all_steps.jsonl", all_steps),):
        with (corruption_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_corruption_records(
        {
            "all_steps": all_steps,
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
