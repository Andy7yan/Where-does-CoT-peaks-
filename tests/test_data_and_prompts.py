"""Tests for GSM8K data preparation and prompt template helpers."""

import json
from pathlib import Path
import shutil
import uuid

from src.data_phase1.gsm8k import (
    build_ranked_questions,
    load_gsm8k_test,
    parse_gold_answer,
    save_gsm8k_corpus,
    save_question_slice,
    slice_question_records,
)
from src.data_phase1.prompting import (
    build_generation_messages,
    build_nldd_clean_prompt,
    build_nldd_corrupt_prompt,
    load_prompt_template,
)


def test_load_gsm8k_test_local_returns_50_records() -> None:
    temp_dir = Path("tests") / f"_tmp_local_gsm8k_{uuid.uuid4().hex}"
    local_path = temp_dir / "gsm8k_sample.json"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            json.dumps(
                [
                    {"question": f"Question {index}", "answer": f"Work it out.\n#### {index}"}
                    for index in range(50)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        records = load_gsm8k_test(source="local", local_path=str(local_path))

        assert len(records) == 50
        assert set(records[0]) == {"question", "answer"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parse_gold_answer_matches_sample_answers(sample_gsm8k: list[dict]) -> None:
    for item in sample_gsm8k[:10]:
        expected = float(item["answer"].split("####", maxsplit=1)[1].strip().splitlines()[0])
        assert parse_gold_answer(item["answer"]) == expected


def test_build_ranked_questions_returns_expected_fields(sample_gsm8k: list[dict]) -> None:
    ranked_questions = build_ranked_questions(sample_gsm8k, hash_seed=42)
    question_slice = slice_question_records(ranked_questions, start_idx=0, end_idx=10)

    assert len(question_slice) == 10
    for index, record in enumerate(question_slice):
        assert record["question_id"] == f"gsm8k_platinum_{index:04d}"
        assert isinstance(record["question_text"], str)
        assert isinstance(record["gold_answer"], float)


def test_build_ranked_questions_is_deterministic(sample_gsm8k: list[dict]) -> None:
    first = build_ranked_questions(sample_gsm8k, hash_seed=42)
    second = build_ranked_questions(sample_gsm8k, hash_seed=42)

    assert list(first) == list(second)


def test_save_question_slice_writes_parseable_jsonl(sample_gsm8k: list[dict]) -> None:
    ranked_questions = build_ranked_questions(sample_gsm8k, hash_seed=42)
    question_slice = slice_question_records(ranked_questions, start_idx=0, end_idx=10)

    temp_dir = Path("tests") / f"_tmp_save_question_slice_{uuid.uuid4().hex}"
    try:
        jsonl_path, meta_path = save_question_slice(question_slice, str(temp_dir))
        lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        assert len(parsed) == 10
        assert metadata["n"] == 10
        assert metadata["hash_seed"] == 42
        assert metadata["start_idx"] == 0
        assert metadata["total_questions"] == 50
        assert metadata["dataset"] == "madrylab/gsm8k-platinum"
        assert metadata["split"] == "test"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_slice_question_records_supports_start_idx_with_global_question_ids(sample_gsm8k: list[dict]) -> None:
    ranked_questions = build_ranked_questions(sample_gsm8k, hash_seed=42)
    question_slice = slice_question_records(ranked_questions, start_idx=10, end_idx=15)

    assert len(question_slice) == 5
    assert [record["question_id"] for record in question_slice] == [
        "gsm8k_platinum_0010",
        "gsm8k_platinum_0011",
        "gsm8k_platinum_0012",
        "gsm8k_platinum_0013",
        "gsm8k_platinum_0014",
    ]


def test_save_question_slice_supports_custom_filename(sample_gsm8k: list[dict]) -> None:
    ranked_questions = build_ranked_questions(sample_gsm8k, hash_seed=42)
    question_slice = slice_question_records(ranked_questions, start_idx=0, end_idx=5)

    temp_dir = Path("tests") / f"_tmp_save_question_slice_named_{uuid.uuid4().hex}"
    try:
        jsonl_path, meta_path = save_question_slice(
            question_slice,
            str(temp_dir),
            jsonl_filename="gsm8k_full_ranked.jsonl",
        )

        assert Path(jsonl_path).name == "gsm8k_full_ranked.jsonl"
        assert Path(meta_path).name == "gsm8k_full_ranked_meta.json"
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
    for prompt_id in (
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ):
        template = load_prompt_template(prompt_id)

        assert template["prompt_id"] == prompt_id
        assert template["version"] == 1
        assert isinstance(template["system"], str)
        assert isinstance(template["few_shot"], list)
        assert len(template["few_shot"]) > 0
        for exemplar in template["few_shot"]:
            assert isinstance(exemplar["user"], str)
            assert isinstance(exemplar["assistant"], str)
        assert template["user_template"] == "{question}"


def test_build_generation_messages_uses_new_signature(sample_question: dict) -> None:
    template = load_prompt_template("icl_short")

    messages = build_generation_messages(
        question=sample_question["question"],
        prompt_template=template,
    )

    assert len(messages) == 1 + 2 * len(template["few_shot"]) + 1
    assert messages[0]["role"] == "system"
    assert "exactly" not in messages[0]["content"].lower()
    assert "target_length" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[1]["content"] == template["few_shot"][0]["user"]
    assert messages[2]["content"] == template["few_shot"][0]["assistant"]
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
