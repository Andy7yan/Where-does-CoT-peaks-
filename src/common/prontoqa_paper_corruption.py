"""Paper-style PrOntoQA step corruption helpers."""

from __future__ import annotations

from collections.abc import Callable
import random
import re

from src.common.corruption import CorruptionResult
from src.common.reasoning import segment_steps


_PAPER_CATEGORIES = [
    "wumpus",
    "gorpus",
    "rompus",
    "jompus",
    "zumpus",
    "tumpus",
    "yumpus",
    "impus",
]

_INFERENCE_RE = re.compile(r"(.*,\s*\w+\s+is\s+a?\s*)(\w+)([\s\.\!\?]*)$", re.IGNORECASE)
_CONCLUSION_RE = re.compile(r"(Conclusion:\s*\w+\s+is\s+a?\s*)(\w+)(,.*)", re.IGNORECASE)


def corrupt_prontoqa_step(
    step_text: str,
    *,
    rng: random.Random | None = None,
    token_counter: Callable[[str], int] | None = None,
    token_delta_max: int = 2,
    retry_limit: int = 3,
    perplexity_scorer: Callable[[str, str], float] | None = None,
    max_perplexity_ratio: float | None = None,
) -> CorruptionResult:
    """Corrupt one logical reasoning step using the paper's PrOntoQA rules."""

    generator = rng or random.Random()
    saw_budget_failure = False
    for attempt_index in range(1, max(retry_limit, 1) + 1):
        candidate_text, corruption_type, original_fragment, corrupted_fragment = (
            _build_candidate(step_text, generator)
        )
        if corruption_type == "uncorruptible":
            return CorruptionResult(
                corrupt_text=step_text,
                original_number="",
                perturbed_number="",
                corruption_failed=True,
                failure_tier="uncorruptible",
            )
        validation = _validate_prontoqa_candidate(
            clean_text=step_text,
            corrupt_text=candidate_text,
            token_counter=token_counter,
            token_delta_max=token_delta_max,
            perplexity_scorer=perplexity_scorer,
            max_perplexity_ratio=max_perplexity_ratio,
        )
        if validation is None:
            saw_budget_failure = True
            continue
        token_delta, perplexity_ratio = validation
        return CorruptionResult(
            corrupt_text=candidate_text,
            original_number=original_fragment,
            perturbed_number=corrupted_fragment,
            corruption_failed=False,
            corruption_tier=3,
            corruption_type=corruption_type,
            original_fragment=original_fragment,
            corrupted_fragment=corrupted_fragment,
            token_delta=token_delta,
            attempts=attempt_index,
            perplexity_ratio=perplexity_ratio,
        )

    return CorruptionResult(
        corrupt_text=step_text,
        original_number="",
        perturbed_number="",
        corruption_failed=True,
        failure_tier="tier3_token_budget" if saw_budget_failure else "uncorruptible",
    )


def _build_candidate(
    step_text: str,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    match_inf = _INFERENCE_RE.search(step_text)
    match_conc = _CONCLUSION_RE.search(step_text)

    if match_inf:
        prefix, inferred_class, suffix = match_inf.groups()
        replacement = _sample_wrong_class(inferred_class, rng)
        return (
            f"{prefix}{replacement}{suffix}",
            "inference_substitution",
            inferred_class,
            replacement,
        )

    if match_conc:
        prefix, inferred_class, suffix = match_conc.groups()
        replacement = _sample_wrong_class(inferred_class, rng)
        return (
            f"{prefix}{replacement}{suffix}",
            "conclusion_substitution",
            inferred_class,
            replacement,
        )

    if " are " in step_text:
        return (
            step_text.replace(" are ", " are not ", 1),
            "rule_negation",
            "are",
            "are not",
        )

    if " is " in step_text:
        return (
            step_text.replace(" is ", " is not ", 1),
            "fact_negation",
            "is",
            "is not",
        )

    return step_text, "uncorruptible", "", ""


def _sample_wrong_class(inferred_class: str, rng: random.Random) -> str:
    wrong_classes = [
        category
        for category in _PAPER_CATEGORIES
        if category.lower() != inferred_class.lower()
    ]
    if not wrong_classes:
        return inferred_class
    return rng.choice(wrong_classes)


def _validate_prontoqa_candidate(
    *,
    clean_text: str,
    corrupt_text: str,
    token_counter: Callable[[str], int] | None,
    token_delta_max: int,
    perplexity_scorer: Callable[[str, str], float] | None,
    max_perplexity_ratio: float | None,
) -> tuple[int, float | None] | None:
    if clean_text == corrupt_text:
        return None
    if not corrupt_text.strip():
        return None
    if segment_steps(corrupt_text).num_steps != 1:
        return None

    token_delta = 0
    if token_counter is not None:
        token_delta = abs(token_counter(clean_text) - token_counter(corrupt_text))
        if token_delta > token_delta_max:
            return None

    perplexity_ratio = None
    if perplexity_scorer is not None and max_perplexity_ratio is not None:
        perplexity_ratio = perplexity_scorer(clean_text, corrupt_text)
        if perplexity_ratio > max_perplexity_ratio:
            return None

    return token_delta, perplexity_ratio
