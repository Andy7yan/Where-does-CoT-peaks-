"""Paper-style synthetic PrOntoQA task helpers."""

from __future__ import annotations

import random
from typing import Any

from src.common.reasoning import ExtractionResult, extract_choice_answer, judge
from src.data_phase1.gsm8k import QuestionSlice


PRONTOQA_PAPER_TASK = "prontoqa_paper"
PRONTOQA_PAPER_QUESTION_ID_PREFIX = "prontoqa_paper"
PRONTOQA_NLDD_SYSTEM_PROMPT = (
    "Solve the logical reasoning problem step by step. "
    "Answer with 'A' or 'B'."
)
PRONTOQA_ANSWER_SUFFIX = "\nFinal Answer:"
PRONTOQA_DEFAULT_MIN_HOPS = 4
PRONTOQA_DEFAULT_MAX_HOPS = 16
PRONTOQA_DEFAULT_QUESTION_COUNT = 1000

_PRONTOQA_ENTITY_NAMES = [
    "Sam",
    "Alex",
    "Riley",
    "Taylor",
    "Morgan",
    "Casey",
    "Jordan",
    "Avery",
    "Wren",
    "Max",
    "Fae",
    "Rex",
]

_PRONTOQA_CATEGORIES = [
    "wumpus",
    "gorpus",
    "rompus",
    "jompus",
    "zumpus",
    "tumpus",
    "yumpus",
    "impus",
    "dumpus",
    "lumpus",
    "numpus",
    "pumpus",
    "grumpus",
    "bumpus",
    "slumpus",
    "clumpus",
    "shumpus",
    "tampus",
    "zimpus",
    "kumpus",
    "fumpus",
    "dimpus",
    "humpus",
    "bampus",
]


def build_synthetic_prontoqa_questions(
    *,
    question_count: int = PRONTOQA_DEFAULT_QUESTION_COUNT,
    min_hops: int = PRONTOQA_DEFAULT_MIN_HOPS,
    max_hops: int = PRONTOQA_DEFAULT_MAX_HOPS,
    hash_seed: int = 42,
    dataset_name: str = PRONTOQA_PAPER_TASK,
    split: str = "synthetic",
) -> QuestionSlice:
    """Build a deterministic synthetic PrOntoQA corpus aligned to the paper."""

    if question_count <= 0:
        raise ValueError("question_count must be positive.")
    if min_hops <= 0:
        raise ValueError("min_hops must be positive.")
    if max_hops < min_hops:
        raise ValueError("max_hops must be greater than or equal to min_hops.")

    rng = random.Random(hash_seed)
    records: list[dict[str, Any]] = []
    for index in range(question_count):
        hops = rng.randint(min_hops, max_hops)
        entity = rng.choice(_PRONTOQA_ENTITY_NAMES)
        actual_len = min(len(_PRONTOQA_CATEGORIES), hops + 2)
        chain = rng.sample(_PRONTOQA_CATEGORIES, actual_len)

        facts = [f"{entity} is a {chain[0]}."]
        rules = [
            f"Every {chain[offset]} is a {chain[offset + 1]}."
            for offset in range(len(chain) - 2)
        ]

        is_reachable_target = bool(rng.randint(0, 1))
        if is_reachable_target:
            target_class = chain[len(rules)]
        else:
            available = [
                category
                for category in _PRONTOQA_CATEGORIES
                if category not in chain[: len(rules) + 1]
            ]
            target_class = rng.choice(available if available else [chain[-1]])

        statement = f"{entity} is a {target_class}."
        gold_answer = "A" if is_reachable_target else "B"

        context = " ".join([*facts, *rules]).strip()
        question_text = (
            f"Context: {context} "
            f"Question: Is the following statement true or false? {statement} "
            "Options: A) True B) False"
        )
        records.append(
            {
                "question_id": f"{PRONTOQA_PAPER_QUESTION_ID_PREFIX}_{index:04d}",
                "question_text": question_text,
                "gold_answer": gold_answer,
            }
        )

    return QuestionSlice(
        records,
        hash_seed=hash_seed,
        start_idx=0,
        total_questions=question_count,
        dataset=dataset_name,
        split=split,
    )


def extract_prontoqa_answer(completion: str) -> ExtractionResult:
    """Extract the final A/B answer for synthetic PrOntoQA traces."""

    return extract_choice_answer(completion)


def judge_prontoqa_answer(
    extracted: float | str | None,
    gold: float | str,
    tolerance: float = 1e-3,
) -> bool:
    """Judge synthetic PrOntoQA answers under the A/B convention."""

    del tolerance
    return judge(extracted, gold)
