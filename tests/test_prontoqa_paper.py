"""Tests for the synthetic PrOntoQA paper-style task integration."""

import json
from pathlib import Path
import shutil
import sys
import uuid

from scripts import run_generation
from scripts.run_generation import discover_prompt_templates
from src.analysis_phase1.nldd import CorruptionSelectionConfig, build_corruption_records
from src.common.prontoqa_paper_corruption import corrupt_prontoqa_step
from src.data_phase1.generation import GenerationOutput, generate_traces_for_question
from src.data_phase1.prontoqa_paper import (
    PRONTOQA_PAPER_TASK,
    build_synthetic_prontoqa_questions,
    extract_prontoqa_answer,
    judge_prontoqa_answer,
)
from src.data_phase1.prompting import build_nldd_clean_prompt, build_nldd_corrupt_prompt


class _FakeProntoGenerator:
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        del messages, temperature, max_new_tokens
        return GenerationOutput(
            raw_completion=(
                "Step 1: Sam is a wumpus.\n"
                "Step 2: Since Sam is a wumpus and every wumpus is a gorpus, Sam is a gorpus.\n"
                "Final Answer: A"
            ),
            token_count=12,
        )


class _FakeProntoBatchGenerator:
    def __init__(self, model_name: str, dtype: str = "float16", cache_dir: str | None = None):
        self.model_name = model_name
        self.dtype = dtype
        self.cache_dir = cache_dir

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list[GenerationOutput]:
        assert temperature == 0.6
        assert max_new_tokens == 1024
        return [
            GenerationOutput(
                raw_completion=(
                    "Step 1: Apply the stated rules.\n"
                    "Step 2: Compare the derived fact with the hypothesis.\n"
                    "Final Answer: A"
                ),
                token_count=9,
            )
            for _ in messages_batch
        ]


def test_prontoqa_synthetic_generator_is_deterministic() -> None:
    first = build_synthetic_prontoqa_questions(question_count=5, min_hops=4, max_hops=6, hash_seed=42)
    second = build_synthetic_prontoqa_questions(question_count=5, min_hops=4, max_hops=6, hash_seed=42)

    assert list(first) == list(second)
    assert len(first) == 5
    assert first[0]["question_id"] == "prontoqa_paper_0000"
    assert "Context:" in first[0]["question_text"]
    assert "Options: A) True B) False" in first[0]["question_text"]
    assert " is not a " not in first[0]["question_text"]
    assert first[0]["gold_answer"] in {"A", "B"}


def test_prontoqa_synthetic_generator_matches_pipeline_positive_query_form() -> None:
    records = build_synthetic_prontoqa_questions(question_count=50, min_hops=4, max_hops=6, hash_seed=42)

    assert all(" is not a " not in row["question_text"] for row in records)
    assert {row["gold_answer"] for row in records} == {"A", "B"}


def test_extract_prontoqa_answer_accepts_ab_and_true_false() -> None:
    assert extract_prontoqa_answer("Final Answer: A").value == "A"
    assert extract_prontoqa_answer("Final Answer: B").value == "B"
    assert extract_prontoqa_answer("Final Answer: True").value == "A"
    assert extract_prontoqa_answer("Final Answer: False").value == "B"
    assert judge_prontoqa_answer("A", "A") is True
    assert judge_prontoqa_answer("B", "A") is False


def test_prontoqa_corruption_hits_inference_substitution() -> None:
    result = corrupt_prontoqa_step(
        "Since Sam is a wumpus and every wumpus is a gorpus, Sam is a gorpus",
        rng=None,
        token_counter=lambda text: len(text.split()),
        token_delta_max=2,
    )

    assert result.corruption_failed is False
    assert result.corruption_type == "inference_substitution"
    assert result.corrupt_text != "Since Sam is a wumpus and every wumpus is a gorpus, Sam is a gorpus"


def test_prontoqa_nldd_prompt_uses_task_header_and_truncation() -> None:
    clean_prompt = build_nldd_clean_prompt(
        question="Context: ... Question: ... Options: A) True B) False",
        steps=["Step 1", "Step 2"],
        task_name=PRONTOQA_PAPER_TASK,
    )
    corrupt_prompt = build_nldd_corrupt_prompt(
        question="Context: ... Question: ... Options: A) True B) False",
        clean_steps=["Step 1", "Step 2", "Step 3"],
        corrupt_step="Bad Step 2",
        corrupt_index=1,
        task_name=PRONTOQA_PAPER_TASK,
    )

    assert clean_prompt.startswith("Solve the logical reasoning problem step by step.")
    assert clean_prompt.endswith("Final Answer: ")
    assert "\nStep 1\nBad Step 2\nFinal Answer: " in corrupt_prompt
    assert "\nStep 3\n" not in corrupt_prompt


def test_generate_traces_for_question_supports_prontoqa_answers() -> None:
    traces = generate_traces_for_question(
        generator=_FakeProntoGenerator(),
        question_id="prontoqa_paper_0001",
        question_text="Context: Sam is a wumpus. Question: Is the following statement true or false? Sam is a gorpus. Options: A) True B) False",
        gold_answer="A",
        prompt_templates=[
            {
                "prompt_id": "icl_short",
                "system": "Solve the user's problem clearly.",
                "few_shot": [],
                "user_template": "{question}",
            }
        ],
        samples_per_group=1,
        temperature=0.6,
        max_new_tokens=64,
        answer_markers=["Final Answer:"],
        answer_extractor=extract_prontoqa_answer,
        answer_judge=judge_prontoqa_answer,
        task_name=PRONTOQA_PAPER_TASK,
    )

    assert len(traces) == 1
    assert traces[0]["task_name"] == PRONTOQA_PAPER_TASK
    assert traces[0]["extracted_answer"] == "A"
    assert traces[0]["is_correct"] is True
    assert traces[0]["final_answer_line"] == "Final Answer: A"


def test_build_corruption_records_supports_prontoqa_task() -> None:
    traces = [
        (
            "root",
            {
                "trace_id": "prontoqa_paper_0000_icl_short_1",
                "task_name": PRONTOQA_PAPER_TASK,
                "question_id": "prontoqa_paper_0000",
                "question_text": "Context: Sam is a wumpus. Question: Is the following statement true or false? Sam is a gorpus. Options: A) True B) False",
                "gold_answer": "A",
                "prompt_id": "icl_short",
                "steps": [
                    "Step 1: Sam is a wumpus.",
                    "Since Sam is a wumpus and every wumpus is a gorpus, Sam is a gorpus",
                ],
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
        task_name=PRONTOQA_PAPER_TASK,
    )

    assert len(records["all_steps"]) == 1
    assert records["all_steps"][0]["task_name"] == PRONTOQA_PAPER_TASK
    assert records["all_steps"][0]["corrupt_prompt"].endswith("Final Answer: ")


def test_discover_prontoqa_prompts_dir_returns_four_templates() -> None:
    templates = discover_prompt_templates(
        prompts_dir="prompts/PrOntoQA",
        expected_count=4,
        preferred_prompt_ids=["icl_short", "icl_medium", "icl_detailed", "icl_verbose"],
    )

    assert [template["prompt_id"] for template in templates] == [
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ]


def test_run_generation_cli_accepts_prontoqa_ab_gold_answers(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setattr(run_generation, "LLMGenerator", _FakeProntoBatchGenerator)

    temp_dir = Path("tests") / f"_tmp_prontoqa_cli_{uuid.uuid4().hex}"
    try:
        output_dir = temp_dir / "run"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_generation.py",
                "--config",
                "configs/stage1_prontoqa.yaml",
                "--output-dir",
                str(output_dir),
                "--shard-id",
                "q0_1",
                "--start-idx",
                "0",
                "--end-idx",
                "1",
            ],
        )

        run_generation.main()

        traces_path = output_dir / "shards" / "q0_1" / "traces.jsonl"
        trace_rows = [
            json.loads(line)
            for line in traces_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(trace_rows) == 20
        assert {row["task_name"] for row in trace_rows} == {PRONTOQA_PAPER_TASK}
        assert {row["gold_answer"] for row in trace_rows} <= {"A", "B"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
