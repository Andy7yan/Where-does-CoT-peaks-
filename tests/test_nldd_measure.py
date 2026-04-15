"""Tests for the v4 Stage D2 NLDD measurement path."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

from src.analysis_phase.nldd import (
    build_correct_token_ids,
    compute_logit_margin,
    extract_trace_horizon,
    measure_nldd,
    measure_trace_profile,
    validate_nldd_full_records,
)
from src.data_phase2.aggregation import aggregate_stage1_outputs


class FakeTokenizer:
    """Minimal tokenizer stub for deterministic unit tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, list[int]]:
        del add_special_tokens, return_tensors
        return {"input_ids": self.encode(text, add_special_tokens=False)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {
            "4": [0],
            " 4": [1],
            "8": [2],
            " 8": [3],
            "12.5": [6, 7],
            " 12.5": [8, 9],
        }
        if text in mapping:
            return list(mapping[text])
        stripped = text.strip()
        if stripped in mapping:
            return list(mapping[stripped])
        return [11]


def test_build_correct_token_ids_handles_integer_and_leading_space() -> None:
    tokenizer = FakeTokenizer()

    token_ids = build_correct_token_ids(4.0, tokenizer)

    assert token_ids == [0, 1]


def test_build_correct_token_ids_uses_first_token_for_multi_token_answers() -> None:
    tokenizer = FakeTokenizer()

    token_ids = build_correct_token_ids(12.5, tokenizer)

    assert token_ids == [6, 8]


def test_compute_logit_margin_and_measure_nldd_handle_basic_cases() -> None:
    margin = compute_logit_margin([4.0, 3.0, 1.5, -1.0], [0, 1], 2.0)

    assert margin == 1.25
    assert round(measure_nldd(1.25, 0.25, ld_epsilon=1.0e-6), 4) == 80.0
    assert measure_nldd(1.0e-8, 0.0, ld_epsilon=1.0e-6) is None


def test_extract_trace_horizon_prefers_smallest_k_on_ties() -> None:
    horizon = extract_trace_horizon(
        [
            {"corruption_step_index": 1, "actual_clean_length": 4, "nldd_value": 9.0},
            {"corruption_step_index": 2, "actual_clean_length": 4, "nldd_value": 5.0},
            {"corruption_step_index": 3, "actual_clean_length": 4, "nldd_value": 5.0},
        ]
    )

    assert horizon["k_star_trace"] == 2
    assert horizon["r_star_trace"] == 0.5


def test_measure_trace_profile_is_deterministic_and_covers_all_steps() -> None:
    tokenizer = FakeTokenizer()
    trace = _trace("q1", 3, True, steps=["2 + 2 = 4", "4 + 0 = 4", "Return 4"])
    selection_row = {
        "trace_id": trace["trace_id"],
        "question_id": trace["question_id"],
        "difficulty": "easy",
        "length_bin": "short",
        "raw_length_bin": "short",
        "selected_for_nldd": True,
        "selected_for_near_lstar": False,
        "selection_mode": "random_sample",
        "near_lstar_selection_mode": None,
    }

    first = measure_trace_profile(
        trace=trace,
        selection_row=selection_row,
        prompt_logits_fn=_fake_prompt_logits,
        tokenizer=tokenizer,
        token_counter=lambda text: len(text.split()),
        s_value=2.0,
        ld_epsilon=1.0e-6,
        seed=42,
        token_delta_max=2,
        retry_limit=3,
    )
    second = measure_trace_profile(
        trace=trace,
        selection_row=selection_row,
        prompt_logits_fn=_fake_prompt_logits,
        tokenizer=tokenizer,
        token_counter=lambda text: len(text.split()),
        s_value=2.0,
        ld_epsilon=1.0e-6,
        seed=42,
        token_delta_max=2,
        retry_limit=3,
    )

    assert [row["corruption_step_index"] for row in first] == [1, 2, 3]
    assert [
        (
            row["corruption_step_index"],
            row["corruption_type"],
            row["corruption_failed"],
            row["nldd_value"],
            row["k_star_trace"],
        )
        for row in first
    ] == [
        (
            row["corruption_step_index"],
            row["corruption_type"],
            row["corruption_failed"],
            row["nldd_value"],
            row["k_star_trace"],
        )
        for row in second
    ]
    assert all(row["k_star_trace"] == first[0]["k_star_trace"] for row in first)
    assert first[0]["k_star_trace"] in {2, 3}


def test_measure_trace_profile_emits_failed_rows_with_null_nldd() -> None:
    tokenizer = FakeTokenizer()
    trace = _trace("q2", 1, True, steps=["There are many apples"])
    selection_row = {
        "trace_id": trace["trace_id"],
        "question_id": trace["question_id"],
        "difficulty": "hard",
        "length_bin": "long",
        "raw_length_bin": "long",
        "selected_for_nldd": True,
        "selected_for_near_lstar": False,
        "selection_mode": "all_available",
        "near_lstar_selection_mode": None,
    }

    rows = measure_trace_profile(
        trace=trace,
        selection_row=selection_row,
        prompt_logits_fn=_fake_prompt_logits,
        tokenizer=tokenizer,
        token_counter=lambda text: len(text.split()),
        s_value=2.0,
        ld_epsilon=1.0e-6,
        seed=42,
        token_delta_max=2,
        retry_limit=3,
    )

    assert len(rows) == 1
    assert rows[0]["corruption_failed"] is True
    assert rows[0]["nldd_value"] is None
    assert rows[0]["measurement_exclusion_reason"] == "corruption_failed"


def test_run_nldd_measure_main_writes_v4_outputs(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_nldd_measure_{uuid.uuid4().hex}"
    run_dir = workspace / "full_generation"
    try:
        _build_stage_c_run(run_dir)

        import scripts.run_nldd_measure as runner

        monkeypatch.setattr(
            runner,
            "load_measurement_backend",
            lambda config: {
                "tokenizer": FakeTokenizer(),
                "prompt_logits_fn": _fake_prompt_logits,
                "runtime_selection": SimpleNamespace(
                    requested_device="cpu",
                    resolved_device="cpu",
                ),
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_nldd_measure.py",
                "--run-dir",
                str(run_dir),
                "--config",
                "configs/stage1.yaml",
            ],
        )

        runner.main()

        selection_path = run_dir / "trace_selection.csv"
        calibration_path = run_dir / "s_calibration.json"
        nldd_path = run_dir / "nldd_full.jsonl"
        summary_path = run_dir / "corruption_summary.json"

        assert selection_path.exists()
        assert calibration_path.exists()
        assert nldd_path.exists()
        assert summary_path.exists()

        selection_rows = runner.load_or_build_trace_selection(
            run_dir=str(run_dir),
            traces=runner.load_stage1_traces(run_dir),
            question_metadata=runner.load_question_metadata(run_dir / "question_metadata.jsonl"),
            coarse_analysis=runner.load_coarse_analysis(run_dir / "coarse_analysis.json"),
            selection_config=runner.TraceSelectionConfig(
                target_traces_per_cell=120,
                target_traces_near_lstar=120,
                per_question_trace_cap=2,
                min_nldd_length=3,
                seed=42,
            ),
        )[0]
        records = [
            json.loads(line)
            for line in nldd_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_nldd_full_records(selection_rows=selection_rows, records=records)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["summary"]["measured_row_count"] == len(records)
        assert not (run_dir / "sampled_steps.jsonl").exists()
        assert not (run_dir / "nldd_spot.jsonl").exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _build_stage_c_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = [
        _trace("q1", 2, True, trace_suffix="a"),
        _trace("q1", 2, True, trace_suffix="b"),
        _trace("q2", 2, True, trace_suffix="a"),
        _trace("q2", 2, False, trace_suffix="b"),
        _trace("q2", 2, True, trace_suffix="c"),
        _trace("q3", 4, False, trace_suffix="a"),
        _trace("q3", 4, True, trace_suffix="b"),
        _trace("q4", 5, True, trace_suffix="a"),
        _trace("q4", 5, True, trace_suffix="b"),
        _trace("q4", 5, True, trace_suffix="c"),
        _trace("q4", 5, True, trace_suffix="d"),
        _trace("q4", 5, True, trace_suffix="e"),
        _trace("q4", 5, False, trace_suffix="f"),
        _trace("q4", 8, False, trace_suffix="g"),
        _trace("q5", 8, False, trace_suffix="a"),
        _trace("q5", 8, True, trace_suffix="b"),
        _trace("q6", 8, False, trace_suffix="a"),
        _trace("q6", 8, False, trace_suffix="b"),
    ]
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")

    (run_dir / "run_meta.json").write_text(
        json.dumps({"schema_version": "stage1_trace_v2"}, indent=2) + "\n",
        encoding="utf-8",
    )
    aggregate_stage1_outputs(str(run_dir))


def _trace(
    question_id: str,
    actual_num_steps: int,
    is_correct: bool,
    *,
    steps: list[str] | None = None,
    trace_suffix: str = "0",
) -> dict[str, object]:
    resolved_steps = steps or [f"2 + 2 = 4 ({index})" for index in range(actual_num_steps)]
    return {
        "trace_id": (
            f"{question_id}_{actual_num_steps}_{int(is_correct)}_{len(resolved_steps)}_{trace_suffix}"
        ),
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": 4.0,
        "prompt_id": "icl_short",
        "raw_completion": "stub",
        "steps": resolved_steps,
        "actual_num_steps": len(resolved_steps),
        "final_answer_line": "#### 4",
        "extracted_answer": 4.0 if is_correct else 0.0,
        "is_correct": is_correct,
        "extraction_failed": False,
        "token_count": 10 + len(resolved_steps),
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _fake_prompt_logits(prompt: str) -> list[float]:
    lowered = prompt.lower()
    if "-" in lowered or "5" in lowered:
        return [0.2, 0.1, 1.6, 1.4, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4]
    return [3.0, 2.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2]
