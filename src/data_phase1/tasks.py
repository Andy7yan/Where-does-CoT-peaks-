"""Task-aware dataset loading and answer-handling helpers for Stage 1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.common.reasoning import DEFAULT_ANSWER_MARKERS, extract_answer, judge
from src.common.settings import ExperimentConfig
from src.data_phase1.gsm8k import build_ranked_questions, load_gsm8k_test
from src.data_phase1.prontoqa_paper import (
    PRONTOQA_ANSWER_SUFFIX,
    PRONTOQA_DEFAULT_MAX_HOPS,
    PRONTOQA_DEFAULT_MIN_HOPS,
    PRONTOQA_DEFAULT_QUESTION_COUNT,
    PRONTOQA_NLDD_SYSTEM_PROMPT,
    PRONTOQA_PAPER_TASK,
    build_synthetic_prontoqa_questions,
    extract_prontoqa_answer,
    judge_prontoqa_answer,
)


GSM8K_TASK = "gsm8k"
GSM8K_NLDD_SYSTEM_PROMPT = (
    "Solve the math problem step by step. Provide your final numerical answer."
)
GSM8K_ANSWER_SUFFIX = "\nFinal Answer:"


def load_question_records_for_config(
    *,
    config: ExperimentConfig,
    source: str = "huggingface",
    local_path: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load ranked question records for the configured task."""

    task_name = get_task_name(config)
    if task_name == PRONTOQA_PAPER_TASK:
        return build_synthetic_prontoqa_questions(
            question_count=(
                config.dataset.synthetic_question_count
                or PRONTOQA_DEFAULT_QUESTION_COUNT
            ),
            min_hops=config.dataset.pronto_min_hops or PRONTOQA_DEFAULT_MIN_HOPS,
            max_hops=config.dataset.pronto_max_hops or PRONTOQA_DEFAULT_MAX_HOPS,
            hash_seed=config.dataset.order_hash_seed,
            dataset_name=config.dataset.name,
            split=config.dataset.split,
        )

    records = load_gsm8k_test(
        source=source,
        local_path=local_path,
        cache_dir=cache_dir,
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.hf_config,
        split=config.dataset.split,
    )
    return build_ranked_questions(
        records,
        hash_seed=config.dataset.order_hash_seed,
        dataset_name=config.dataset.name,
        split=config.dataset.split,
    )


def get_task_name(config: ExperimentConfig) -> str:
    """Return the normalized task name for the active config."""

    task_name = str(config.dataset.task).strip().lower()
    if not task_name:
        return GSM8K_TASK
    return task_name


def get_prompts_dir(config: ExperimentConfig) -> str:
    """Return the prompt directory for the configured task."""

    if config.dataset.prompts_dir:
        return config.dataset.prompts_dir
    if get_task_name(config) == PRONTOQA_PAPER_TASK:
        return "prompts/PrOntoQA"
    return "prompts/GSM8k"


def get_answer_extractor(
    config: ExperimentConfig,
) -> Callable[[str], Any]:
    """Return the completion -> extracted answer callable for the task."""

    if config.answer_extraction.mode == "choice_ab":
        return extract_prontoqa_answer
    return extract_answer


def get_answer_judge(
    config: ExperimentConfig,
) -> Callable[[float | str | None, float | str, float], bool]:
    """Return the task-aware answer-judging callable."""

    if config.answer_extraction.mode == "choice_ab":
        return judge_prontoqa_answer
    return judge


def get_answer_markers(config: ExperimentConfig) -> list[str]:
    """Return the step-segmentation answer markers for the task."""

    markers = list(config.step_segmentation.answer_markers)
    if markers:
        return markers
    if config.answer_extraction.mode == "choice_ab":
        return ["Final Answer:"]
    return list(DEFAULT_ANSWER_MARKERS)


def get_nldd_system_prompt(task_name: str) -> str:
    """Return the task-specific NLDD prompt header."""

    if task_name == PRONTOQA_PAPER_TASK:
        return PRONTOQA_NLDD_SYSTEM_PROMPT
    return GSM8K_NLDD_SYSTEM_PROMPT


def get_answer_suffix(task_name: str) -> str:
    """Return the task-specific answer cue used in NLDD prompts."""

    if task_name == PRONTOQA_PAPER_TASK:
        return PRONTOQA_ANSWER_SUFFIX
    return GSM8K_ANSWER_SUFFIX

