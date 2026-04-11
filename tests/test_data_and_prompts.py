"""Tests for GSM8K data preparation and prompt template helpers."""

import json
from pathlib import Path
import shutil
import uuid

from src.gsm8k import (
    load_gsm8k_test,
    parse_gold_answer,
    save_eval_subset,
    save_gsm8k_corpus,
    select_eval_subset,
)
from src.prompting import (
    build_generation_messages,
    build_nldd_clean_prompt,
    build_nldd_corrupt_prompt,
    load_prompt_template,
)


def test_load_gsm8k_test_local_returns_50_records() -> None:
    records = load_gsm8k_test(source="local", local_path="data/gsm8k_sample.json")

    assert len(records) == 50
    assert set(records[0]) == {"question", "answer"}


def test_parse_gold_answer_matches_sample_answers(sample_gsm8k: list[dict]) -> None:
    for item in sample_gsm8k[:10]:
        expected = float(item["answer"].split("####", maxsplit=1)[1].strip().splitlines()[0])
        assert parse_gold_answer(item["answer"]) == expected


def test_select_eval_subset_returns_expected_fields(sample_gsm8k: list[dict]) -> None:
    subset = select_eval_subset(sample_gsm8k, n=10, hash_seed=42)

    assert len(subset) == 10
    for index, record in enumerate(subset):
        assert record["question_id"] == f"gsm8k_{index:04d}"
        assert isinstance(record["question_text"], str)
        assert isinstance(record["gold_answer"], float)


def test_select_eval_subset_is_deterministic(sample_gsm8k: list[dict]) -> None:
    first = select_eval_subset(sample_gsm8k, n=10, hash_seed=42)
    second = select_eval_subset(sample_gsm8k, n=10, hash_seed=42)

    assert list(first) == list(second)


def test_save_eval_subset_writes_parseable_jsonl(sample_gsm8k: list[dict]) -> None:
    subset = select_eval_subset(sample_gsm8k, n=10, hash_seed=42)

    temp_dir = Path("tests") / f"_tmp_save_eval_subset_{uuid.uuid4().hex}"
    try:
        jsonl_path, meta_path = save_eval_subset(subset, str(temp_dir))
        lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        assert len(parsed) == 10
        assert metadata["n"] == 10
        assert metadata["hash_seed"] == 42
        assert metadata["dataset"] == "gsm8k"
        assert metadata["split"] == "test"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_gsm8k_corpus_writes_raw_jsonl(sample_gsm8k: list[dict]) -> None:
    temp_dir = Path("tests") / f"_tmp_save_gsm8k_corpus_{uuid.uuid4().hex}"
    try:
        corpus_path = save_gsm8k_corpus(sample_gsm8k[:3], str(temp_dir))
        lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]

        assert len(parsed) == 3
        assert parsed[0]["question"] == sample_gsm8k[0]["question"]
        assert parsed[0]["answer"] == sample_gsm8k[0]["answer"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_icl_prompt_templates_have_expected_schema() -> None:
    for prompt_id in ("icl_short", "icl_medium", "icl_detailed"):
        template = load_prompt_template(prompt_id)

        assert template["prompt_id"] == prompt_id
        assert template["version"] == 1
        assert isinstance(template["system"], str)
        assert isinstance(template["few_shot"], list)
        assert template["few_shot"] == []
        assert template["user_template"] == "{question}"


def test_build_generation_messages_uses_new_signature(sample_question: dict) -> None:
    template = load_prompt_template("icl_short")

    messages = build_generation_messages(
        question=sample_question["question"],
        prompt_template=template,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "exactly" not in messages[0]["content"].lower()
    assert "target_length" not in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == sample_question["question"]


def test_build_generation_messages_rejects_target_length_kwarg(sample_question: dict) -> None:
    template = load_prompt_template("icl_short")

    try:
        build_generation_messages(
            question=sample_question["question"],
            target_length=5,
            prompt_template=template,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("build_generation_messages should reject target_length")


def test_build_nldd_clean_prompt_has_answer_suffix_space() -> None:
    prompt = build_nldd_clean_prompt(
        question="What is 2 + 2?",
        steps=["Add 2 and 2 to get 4."],
    )

    assert prompt.endswith("#### ")
    assert "Step 1: Add 2 and 2 to get 4." in prompt


def test_build_nldd_corrupt_prompt_truncates_after_corrupt_index() -> None:
    prompt = build_nldd_corrupt_prompt(
        question="Question?",
        clean_steps=["A", "B", "C"],
        corrupt_step="X",
        corrupt_index=1,
    )

    assert "Step 1: A" in prompt
    assert "Step 2: X" in prompt
    assert "Step 3" not in prompt


def test_build_nldd_corrupt_prompt_from_first_step() -> None:
    prompt = build_nldd_corrupt_prompt(
        question="Question?",
        clean_steps=["A", "B", "C"],
        corrupt_step="X",
        corrupt_index=0,
    )

    assert "Step 1: X" in prompt
    assert "Step 2" not in prompt
