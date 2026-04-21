"""Tests for NLDD measurement helpers."""

from __future__ import annotations

import json
from pathlib import Path
from src.analysis_phase1.nldd import (
    build_correct_token_ids,
    compute_logit_margin,
    extract_trace_horizon,
    measure_nldd,
    measure_trace_profile,
    validate_nldd_full_records,
)
from src.analysis_phase1.nldd_shared import _move_model_inputs_to_device
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


class FakeBatchEncoding:
    """Small BatchEncoding-like stub for tokenizer output compatibility tests."""

    def __init__(self, payload: dict[str, list[int]]) -> None:
        self._payload = payload

    def to(self, device: str) -> "FakeBatchEncoding":
        del device
        return self

    def items(self):
        return self._payload.items()


class FakeBatchEncodingTokenizer(FakeTokenizer):
    """Tokenizer stub that returns BatchEncoding-like objects instead of dicts."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> FakeBatchEncoding:
        del add_special_tokens, return_tensors
        return FakeBatchEncoding({"input_ids": self.encode(text, add_special_tokens=False)})


def test_build_correct_token_ids_handles_integer_and_leading_space() -> None:
    tokenizer = FakeTokenizer()

    token_ids = build_correct_token_ids(4.0, tokenizer)

    assert token_ids == [0, 1]


def test_build_correct_token_ids_uses_first_token_for_multi_token_answers() -> None:
    tokenizer = FakeTokenizer()

    token_ids = build_correct_token_ids(12.5, tokenizer)

    assert token_ids == [6, 8]


def test_build_correct_token_ids_accepts_batch_encoding_like_tokenizer_output() -> None:
    tokenizer = FakeBatchEncodingTokenizer()

    token_ids = build_correct_token_ids(4.0, tokenizer)

    assert token_ids == [0, 1]


def test_compute_logit_margin_and_measure_nldd_handle_basic_cases() -> None:
    margin = compute_logit_margin([4.0, 3.0, 1.5, -1.0], [0, 1], 2.0)

    assert margin == 1.25
    assert round(measure_nldd(1.25, 0.25, ld_epsilon=1.0e-6), 4) == 80.0
    assert measure_nldd(1.0e-8, 0.0, ld_epsilon=1.0e-6) is None


def test_move_model_inputs_to_device_accepts_batch_encoding_like_objects() -> None:
    moved = _move_model_inputs_to_device(
        FakeBatchEncoding({"input_ids": [1, 2], "attention_mask": [1, 1]}),
        device="cuda",
    )

    assert moved == {"input_ids": [1, 2], "attention_mask": [1, 1]}


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


def test_measure_trace_profile_is_deterministic_and_covers_k_from_two_onward() -> None:
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

    assert [row["corruption_step_index"] for row in first] == [2, 3]
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
    trace = _trace("q2", 2, True, steps=["2 + 2 = 4", "There are many apples"])
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
    assert rows[0]["corruption_step_index"] == 2
    assert rows[0]["corruption_failed"] is True
    assert rows[0]["nldd_value"] is None
    assert rows[0]["measurement_exclusion_reason"] == "corruption_failed"


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


def _fake_trace_trajectory(question: str, steps: tuple[str, ...]) -> list[list[float]]:
    vectors: list[list[float]] = [[0.0, float(len(question)), 0.0]]
    running_chars = len(question)
    for index, step in enumerate(steps, start=1):
        running_chars += len(step)
        vectors.append([float(index), float(running_chars), float(len(step))])
    return vectors
