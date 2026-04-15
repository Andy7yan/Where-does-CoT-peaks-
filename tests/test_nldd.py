"""Tests for corruption-regeneration helpers."""

from src.analysis_phase.nldd import (
    CorruptionSelectionConfig,
    build_corruption_records,
    sample_step_indices_for_trace,
)


def test_sample_step_indices_for_trace_is_deterministic() -> None:
    first = sample_step_indices_for_trace(
        trace_id="trace-1",
        num_steps=5,
        sampled_min_steps=1,
        sampled_max_steps=2,
        seed=42,
    )
    second = sample_step_indices_for_trace(
        trace_id="trace-1",
        num_steps=5,
        sampled_min_steps=1,
        sampled_max_steps=2,
        seed=42,
    )

    assert first == second
    assert 1 <= len(first) <= 2


def test_build_corruption_records_emits_all_and_sampled_modes() -> None:
    traces = [
        (
            "q0000_0001",
            {
                "trace_id": "gsm8k_0000_icl_short_1",
                "question_id": "gsm8k_0000",
                "question_text": "What is 2 + 2?",
                "prompt_id": "icl_short",
                "steps": ["2 + 2 = 4", "So the answer is 4."],
                "actual_num_steps": 2,
                "is_correct": True,
            },
        )
    ]
    records = build_corruption_records(
        traces,
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        retry_limit=3,
        selection=CorruptionSelectionConfig(sampled_min_steps=1, sampled_max_steps=2, seed=42),
    )

    assert len(records["all_steps"]) == 2
    assert 1 <= len(records["sampled_steps"]) <= 2
    assert all("corruption_tier" in record for record in records["all_steps"])


def test_build_corruption_records_disables_tier3_by_default() -> None:
    traces = [
        (
            "q0000_0001",
            {
                "trace_id": "gsm8k_0000_icl_short_1",
                "question_id": "gsm8k_0000",
                "question_text": "Question?",
                "prompt_id": "icl_short",
                "steps": ["The price increased after the sale."],
                "actual_num_steps": 1,
                "is_correct": True,
            },
        )
    ]
    records = build_corruption_records(
        traces,
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        retry_limit=3,
        selection=CorruptionSelectionConfig(seed=42),
    )

    assert records["all_steps"][0]["corruption_failed"] is True
    assert records["all_steps"][0]["failure_tier"] == "uncorruptible"
