"""Tests for corruption-regeneration helpers."""

from src.analysis_phase1.nldd import (
    CorruptionSelectionConfig,
    build_corruption_records,
)


def test_build_corruption_records_emits_full_mode() -> None:
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
        selection=CorruptionSelectionConfig(seed=42),
    )

    assert len(records["all_steps"]) == 2
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
