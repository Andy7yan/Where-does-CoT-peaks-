"""Tests for the project skeleton and Stage 1 core utilities."""

import random
import re

from src.reasoning import (
    corrupt_arithmetic,
    corrupt_step_text_with_fallbacks,
    extract_answer,
    judge,
    normalize_numeric,
    segment_steps,
)
from src.settings import ExperimentConfig, load_settings, require_config_value


def test_sample_question_fixture(sample_question: dict) -> None:
    assert "question" in sample_question
    assert "answer" in sample_question


def test_sample_dataset_has_expected_size(sample_gsm8k: list[dict]) -> None:
    assert len(sample_gsm8k) == 50


def test_config_can_be_loaded(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "test")

    config = ExperimentConfig.from_yaml("configs/stage1.yaml")

    assert config.experiment.run_id == "peak-cot-stage1-gsm8k-llama31"
    assert config.model.hf_cache == "/tmp/hf-home/hub"
    assert config.answer_extraction.numeric_tolerance == 1e-3
    assert config.output.base_dir == "/tmp/runs/test"
    assert config.dataset.subset_size == 200
    assert config.generation.num_icl_groups == 5
    assert config.generation.samples_per_group == 3
    assert config.generation.temperature == 0.7
    assert config.generation.icl_group_prompt_ids == [
        "icl_minimal",
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ]
    assert config.generation.icl_group_temperatures == {}
    assert config.generation.icl_group_sample_counts == {
        "icl_minimal": 3,
        "icl_short": 3,
        "icl_medium": 3,
        "icl_detailed": 5,
        "icl_verbose": 5,
    }
    assert config.generation.max_new_tokens == 544
    assert config.pilot.num_questions == 50
    assert config.tas.layer == "middle"
    assert config.tas.plateau_threshold == 0.05
    assert config.analysis.min_bin_size == 5
    assert config.analysis.num_full_analysis_questions == 25
    assert config.analysis.num_spot_checks == 3
    assert config.analysis.max_extraction_fail_rate == 0.05


def test_load_settings_keeps_null_pilot_fields(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "stage-a")

    settings = load_settings("configs/stage1.yaml")

    assert settings["experiment"]["run_id"] == "peak-cot-stage1-gsm8k-llama31"
    assert settings["dataset"]["subset_size"] == 200
    assert settings["generation"]["num_icl_groups"] == 5
    assert settings["generation"]["samples_per_group"] == 3
    assert settings["generation"]["temperature"] == 0.7
    assert settings["generation"]["icl_groups"] == {
        "icl_minimal": {"samples_per_group": 3},
        "icl_short": {"samples_per_group": 3},
        "icl_medium": {"samples_per_group": 3},
        "icl_detailed": {"samples_per_group": 5},
        "icl_verbose": {"samples_per_group": 5},
    }
    assert settings["generation"]["max_new_tokens"] == 544
    assert settings["tas"]["plateau_threshold"] == 0.05
    assert settings["analysis"]["min_bin_size"] == 5
    assert settings["analysis"]["num_full_analysis_questions"] == 25
    assert settings["analysis"]["num_spot_checks"] == 3
    assert settings["analysis"]["max_extraction_fail_rate"] == 0.05
    assert settings["output"]["base_dir"] == "/tmp/runs/stage-a"


def test_require_config_value_returns_value() -> None:
    assert require_config_value("generation.temperature", 0.7) == 0.7


def test_require_config_value_raises_for_none() -> None:
    try:
        require_config_value("generation.temperature", None)
    except ValueError as exc:
        assert str(exc) == "generation.temperature 需由 Pilot Run 确认后填写"
    else:
        raise AssertionError("require_config_value should raise when the value is None")


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


def test_corrupt_arithmetic_prefers_rhs_result_without_min_threshold() -> None:
    result = corrupt_arithmetic("So 5 * 3 = 15", rng=random.Random(42))

    assert result.corruption_failed is False
    assert result.original_number == "15"
    assert result.perturbed_number != "15"
    assert result.corruption_type == "numeric_result"
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
    assert result.failure_tier == "uncorruptible"


def test_corrupt_arithmetic_is_deterministic_with_fixed_seed() -> None:
    first = corrupt_arithmetic("Total is 250 units", rng=random.Random(42))
    second = corrupt_arithmetic("Total is 250 units", rng=random.Random(42))

    assert first == second


def test_corrupt_arithmetic_multiplier_stays_outside_exclusion_zone() -> None:
    for seed in range(20):
        result = corrupt_arithmetic("Value: 100.0", rng=random.Random(seed))
        perturbed = normalize_numeric(result.perturbed_number)
        assert perturbed is not None
        ratio = perturbed / 100.0
        assert ratio <= 0.9 or ratio >= 1.1


def test_corrupt_arithmetic_on_sample_step(sample_gsm8k: list[dict]) -> None:
    step_text = ""
    for item in sample_gsm8k:
        for candidate in segment_steps(item["answer"]).steps:
            if re.search(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", candidate):
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


def test_corrupt_step_text_with_fallbacks_swaps_operator() -> None:
    result = corrupt_step_text_with_fallbacks(
        "Let x + y be the total.",
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        rng=random.Random(42),
    )

    assert result.corruption_failed is False
    assert result.corruption_tier == 2
    assert result.corruption_type == "operator_swap"


def test_corrupt_step_text_with_fallbacks_uses_semantic_flip() -> None:
    result = corrupt_step_text_with_fallbacks(
        "The price increased after the sale.",
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        rng=random.Random(42),
        use_tier3=True,
    )

    assert result.corruption_failed is False
    assert result.corruption_tier == 3
    assert result.corruption_type == "semantic_flip"


def test_corrupt_step_text_with_fallbacks_marks_uncorruptible() -> None:
    result = corrupt_step_text_with_fallbacks(
        "Therefore x equals y.",
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        rng=random.Random(42),
    )

    assert result.corruption_failed is True
    assert result.failure_tier == "uncorruptible"


def test_corrupt_step_text_with_fallbacks_disables_semantic_flip_by_default() -> None:
    result = corrupt_step_text_with_fallbacks(
        "The price increased after the sale.",
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
        rng=random.Random(42),
    )

    assert result.corruption_failed is True
    assert result.failure_tier == "uncorruptible"
