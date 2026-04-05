"""Tests for the project skeleton and Stage 1 core utilities."""

from __future__ import annotations

import random
import re

from src.core.answer_extraction import extract_answer, judge, normalize_numeric
from src.core.corruption import corrupt_arithmetic
from src.core.step_segmentation import segment_steps
from src.config import ExperimentConfig


def test_sample_question_fixture(sample_question: dict) -> None:
    assert "question" in sample_question
    assert "answer" in sample_question


def test_sample_dataset_has_expected_size(sample_gsm8k: list[dict]) -> None:
    assert len(sample_gsm8k) == 50


def test_config_can_be_loaded(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "test")

    config = ExperimentConfig.from_yaml("configs/stage1.yaml")

    assert config.experiment.name == "peak-cot-stage1-gsm8k-llama31"
    assert config.model.hf_cache == "/tmp/hf-home/hub"
    assert config.answer_extraction.numeric_tolerance == 1e-3
    assert config.output.base_dir == "/tmp/runs/test"


def test_segment_steps_with_answer_line() -> None:
    completion = (
        "First compute 2 + 2 = 4.\n"
        "Then multiply by 3 to get 12.\n"
        "#### 12\n"
    )

    result = segment_steps(completion)

    assert result.steps == [
        "First compute 2 + 2 = 4.",
        "Then multiply by 3 to get 12.",
    ]
    assert result.final_answer_line == "#### 12"
    assert result.num_steps == 2


def test_segment_steps_discards_blank_and_punctuation_only_lines() -> None:
    completion = "\n...\n Step 1: add 5 and 6.\n \n!!!\nThe answer is 11\n"

    result = segment_steps(completion)

    assert result.steps == ["Step 1: add 5 and 6."]
    assert result.final_answer_line == "The answer is 11"
    assert result.num_steps == 1


def test_segment_steps_without_answer_line() -> None:
    result = segment_steps("Plan the work.\nCheck the result.")

    assert result.steps == ["Plan the work.", "Check the result."]
    assert result.final_answer_line is None
    assert result.num_steps == 2


def test_segment_steps_empty_input() -> None:
    result = segment_steps("")

    assert result.steps == []
    assert result.final_answer_line is None
    assert result.num_steps == 0


def test_segment_steps_handles_sample_answers(sample_gsm8k: list[dict]) -> None:
    for item in sample_gsm8k[:5]:
        result = segment_steps(item["answer"])
        assert result.num_steps > 0
        assert result.final_answer_line is not None
        assert "####" in result.final_answer_line


def test_extract_answer_from_hash_marker() -> None:
    result = extract_answer("#### 42")

    assert result.value == 42.0
    assert result.raw_match == "42"
    assert result.extraction_failed is False


def test_extract_answer_with_currency_and_commas() -> None:
    result = extract_answer("#### $1,234.56")

    assert result.value == 1234.56
    assert result.raw_match == "1,234.56"
    assert result.extraction_failed is False


def test_extract_answer_from_phrase_marker() -> None:
    result = extract_answer("The answer is 100.")

    assert result.value == 100.0
    assert result.raw_match == "100."
    assert result.extraction_failed is False


def test_extract_answer_failure() -> None:
    result = extract_answer("blah blah blah")

    assert result.value is None
    assert result.raw_match is None
    assert result.extraction_failed is True


def test_extract_answer_uses_first_marker_match() -> None:
    result = extract_answer("Step 3: #### 10\n#### 20")

    assert result.value == 10.0
    assert result.raw_match == "10"
    assert result.extraction_failed is False


def test_normalize_numeric_strips_formatting() -> None:
    assert normalize_numeric(" $1,234.56% ") == 1234.56


def test_judge_true_within_tolerance() -> None:
    assert judge(42.0, 42.0001) is True


def test_judge_false_for_missing_value() -> None:
    assert judge(None, 42.0) is False


def test_extract_answer_handles_sample_answers(sample_gsm8k: list[dict]) -> None:
    for item in sample_gsm8k[:10]:
        result = extract_answer(item["answer"])
        expected_raw = item["answer"].split("####", maxsplit=1)[1].strip().splitlines()[0]
        expected_value = normalize_numeric(expected_raw)
        assert expected_value is not None
        assert result.extraction_failed is False
        assert result.value == expected_value


def test_corrupt_arithmetic_prefers_value_above_threshold() -> None:
    result = corrupt_arithmetic("So 5 * 3 = 15", rng=random.Random(42))

    assert result.corruption_failed is False
    assert result.original_number == "15"
    assert result.perturbed_number != "15"
    assert result.corrupt_text == f"So 5 * 3 = {result.perturbed_number}"


def test_corrupt_arithmetic_handles_formatted_numbers() -> None:
    result = corrupt_arithmetic("Total: $1,200", rng=random.Random(42))

    assert result.corruption_failed is False
    assert result.original_number == "1,200"
    assert result.perturbed_number != "1,200"
    assert result.corrupt_text.startswith("Total: $")


def test_corrupt_arithmetic_failure_when_no_numeric_candidate() -> None:
    result = corrupt_arithmetic("Therefore x = y")

    assert result.corruption_failed is True
    assert result.corrupt_text == "Therefore x = y"
    assert result.original_number == ""
    assert result.perturbed_number == ""


def test_corrupt_arithmetic_is_deterministic_with_fixed_seed() -> None:
    first = corrupt_arithmetic("Total is 250 units", rng=random.Random(42))
    second = corrupt_arithmetic("Total is 250 units", rng=random.Random(42))

    assert first == second


def test_corrupt_arithmetic_multiplier_stays_outside_exclusion_zone() -> None:
    for seed in range(20):
        result = corrupt_arithmetic("Value: 100", rng=random.Random(seed))
        perturbed = normalize_numeric(result.perturbed_number)
        assert perturbed is not None
        ratio = perturbed / 100.0
        assert ratio <= 0.9 or ratio >= 1.1


def test_corrupt_arithmetic_on_sample_step(sample_gsm8k: list[dict]) -> None:
    step_text = ""
    for item in sample_gsm8k:
        for candidate in segment_steps(item["answer"]).steps:
            if any(
                abs(float(match.replace(",", ""))) >= 10
                for match in re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", candidate)
            ):
                step_text = candidate
                break
        if step_text:
            break

    assert step_text

    result = corrupt_arithmetic(step_text, rng=random.Random(42))

    assert result.corruption_failed is False
    assert result.corrupt_text != step_text
    assert result.original_number in step_text
    assert result.perturbed_number in result.corrupt_text
    assert len(re.findall(r"=", step_text)) == len(re.findall(r"=", result.corrupt_text))
