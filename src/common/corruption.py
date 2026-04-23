"""Paper-aligned step-level corruption utilities for NLDD workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Callable

from src.common.reasoning import normalize_numeric, segment_steps


ARITHMETIC_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
DEFAULT_INTEGER_PERTURBATION_RANGE = (-1, 1)
DEFAULT_FLOAT_PERTURBATION_RANGE = (0.1, 0.5, 2.0, 10.0)
OPERATOR_CHARS = {"+", "-", "*", "/"}
SEMANTIC_FLIP_PAIRS = [
    ("more", "less"),
    ("fewer", "more"),
    ("less", "more"),
    ("gained", "lost"),
    ("lost", "gained"),
    ("increased", "decreased"),
    ("decreased", "increased"),
    ("added", "removed"),
    ("removed", "added"),
    ("bought", "sold"),
    ("sold", "bought"),
    ("gives", "takes"),
    ("takes", "gives"),
]


@dataclass(frozen=True)
class CorruptionResult:
    """Structured result for one corruption attempt."""

    corrupt_text: str
    original_number: str
    perturbed_number: str
    corruption_failed: bool
    corruption_tier: int = 4
    corruption_type: str = "uncorruptible"
    original_fragment: str = ""
    corrupted_fragment: str = ""
    token_delta: int = 0
    attempts: int = 0
    perplexity_ratio: float | None = None
    failure_tier: str | None = None


@dataclass(frozen=True)
class _NumberMatch:
    """Internal representation of a replaceable numeric span."""

    start: int
    end: int
    raw: str
    value: float


def corrupt_arithmetic(
    step_text: str,
    *,
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, ...] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    rng: random.Random | None = None,
    min_value: float = 0.0,
    token_counter: Callable[[str], int] | None = None,
    token_delta_max: int | None = None,
    retry_limit: int = 3,
    perplexity_scorer: Callable[[str, str], float] | None = None,
    max_perplexity_ratio: float | None = None,
) -> CorruptionResult:
    """Apply the paper-style numeric fallback to the last number in the step."""

    del integer_perturbation_range  # Kept for config compatibility.

    generator = rng or random.Random()
    matches = [
        match
        for match in _find_numeric_matches(step_text)
        if abs(match.value) >= min_value
    ]
    if not matches:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
            failure_tier="uncorruptible",
        )

    selected = matches[-1]
    saw_budget_failure = False
    for attempt_index in range(1, max(retry_limit, 1) + 1):
        multiplier = _sample_multiplier(
            float_perturbation_range=float_perturbation_range,
            rng=generator,
        )
        perturbed_number = _format_pipeline_perturbed_value(
            selected.raw,
            selected.value,
            multiplier,
        )
        if perturbed_number == selected.raw:
            continue
        corrupt_text = _replace_fragment(
            step_text,
            selected.start,
            selected.end,
            perturbed_number,
        )
        validation = _validate_corruption_candidate(
            clean_text=step_text,
            corrupt_text=corrupt_text,
            expected_change="numeric",
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
            corrupt_text=corrupt_text,
            original_number=selected.raw,
            perturbed_number=perturbed_number,
            corruption_failed=False,
            corruption_tier=1,
            corruption_type="perturbation",
            original_fragment=selected.raw,
            corrupted_fragment=perturbed_number,
            token_delta=token_delta,
            attempts=attempt_index,
            perplexity_ratio=perplexity_ratio,
        )

    return CorruptionResult(
        corrupt_text=step_text,
        original_number="",
        perturbed_number="",
        corruption_failed=True,
        failure_tier="tier1_token_budget" if saw_budget_failure else "uncorruptible",
    )


def corrupt_step_text_with_fallbacks(
    step_text: str,
    *,
    rng: random.Random | None = None,
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, ...] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
    token_counter: Callable[[str], int] | None = None,
    token_delta_max: int = 2,
    retry_limit: int = 3,
    use_tier3: bool | None = None,
    perplexity_scorer: Callable[[str, str], float] | None = None,
    max_perplexity_ratio: float | None = None,
) -> CorruptionResult:
    """Apply the paper-aligned order: operator swap, then numeric fallback."""

    generator = rng or random.Random()
    tier3_enabled = enable_tier3_semantic_flip or bool(use_tier3)

    operator_swap = _corrupt_operator(
        step_text,
        rng=generator,
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        perplexity_scorer=perplexity_scorer,
        max_perplexity_ratio=max_perplexity_ratio,
    )
    if not operator_swap.corruption_failed:
        return operator_swap

    numeric_fallback = corrupt_arithmetic(
        step_text,
        integer_perturbation_range=integer_perturbation_range,
        float_perturbation_range=float_perturbation_range,
        rng=generator,
        min_value=0.0,
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        perplexity_scorer=perplexity_scorer,
        max_perplexity_ratio=max_perplexity_ratio,
    )
    if not numeric_fallback.corruption_failed:
        return numeric_fallback

    tier3_failure_tier: str | None = None
    if tier3_enabled:
        tier3 = _corrupt_semantic_flip(
            step_text,
            token_counter=token_counter,
            token_delta_max=token_delta_max,
            perplexity_scorer=perplexity_scorer,
            max_perplexity_ratio=max_perplexity_ratio,
        )
        if not tier3.corruption_failed:
            return tier3
        tier3_failure_tier = tier3.failure_tier

    return CorruptionResult(
        corrupt_text=step_text,
        original_number="",
        perturbed_number="",
        corruption_failed=True,
        failure_tier=(
            operator_swap.failure_tier
            or numeric_fallback.failure_tier
            or tier3_failure_tier
            or "uncorruptible"
        ),
    )


def corrupt_step_text(step_text: str) -> str | None:
    """Backward-compatible wrapper around the numeric fallback."""

    result = corrupt_arithmetic(step_text)
    if result.corruption_failed:
        return None
    return result.corrupt_text


def _find_numeric_matches(text: str) -> list[_NumberMatch]:
    matches: list[_NumberMatch] = []
    for match in ARITHMETIC_NUMBER_RE.finditer(text):
        raw = match.group(0)
        value = normalize_numeric(raw)
        if value is None:
            continue
        matches.append(
            _NumberMatch(
                start=match.start(),
                end=match.end(),
                raw=raw,
                value=value,
            )
        )
    return matches


def _replace_fragment(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _validate_corruption_candidate(
    *,
    clean_text: str,
    corrupt_text: str,
    expected_change: str,
    token_counter: Callable[[str], int] | None,
    token_delta_max: int | None,
    perplexity_scorer: Callable[[str, str], float] | None,
    max_perplexity_ratio: float | None,
) -> tuple[int, float | None] | None:
    if clean_text == corrupt_text:
        return None
    if not corrupt_text.strip():
        return None
    if segment_steps(corrupt_text).num_steps != 1:
        return None
    if not _has_legal_numeric_tokens(corrupt_text):
        return None
    if expected_change == "numeric":
        if _count_changed_numeric_spans(clean_text, corrupt_text) != 1:
            return None
        if _numeric_template(clean_text) != _numeric_template(corrupt_text):
            return None
    elif expected_change == "operator":
        if _operator_change_count(clean_text, corrupt_text) != 1:
            return None
        if _extract_numeric_spans(clean_text) != _extract_numeric_spans(corrupt_text):
            return None

    token_delta = 0
    if token_counter is not None and token_delta_max is not None:
        token_delta = abs(token_counter(clean_text) - token_counter(corrupt_text))
        if token_delta > token_delta_max:
            return None

    perplexity_ratio = None
    if perplexity_scorer is not None and max_perplexity_ratio is not None:
        perplexity_ratio = perplexity_scorer(clean_text, corrupt_text)
        if perplexity_ratio > max_perplexity_ratio:
            return None

    return token_delta, perplexity_ratio


def _numeric_template(text: str) -> str:
    return ARITHMETIC_NUMBER_RE.sub("<NUM>", text)


def _count_changed_numeric_spans(clean_text: str, corrupt_text: str) -> int:
    clean_numbers = _extract_numeric_spans(clean_text)
    corrupt_numbers = _extract_numeric_spans(corrupt_text)
    if len(clean_numbers) != len(corrupt_numbers):
        return -1
    return sum(1 for clean, corrupt in zip(clean_numbers, corrupt_numbers) if clean != corrupt)


def _operator_change_count(clean_text: str, corrupt_text: str) -> int:
    if len(clean_text) != len(corrupt_text):
        return -1
    return sum(1 for clean, corrupt in zip(clean_text, corrupt_text) if clean != corrupt)


def _extract_numeric_spans(text: str) -> list[str]:
    return [match.group(0) for match in ARITHMETIC_NUMBER_RE.finditer(text)]


def _has_legal_numeric_tokens(text: str) -> bool:
    for match in ARITHMETIC_NUMBER_RE.finditer(text):
        normalized = normalize_numeric(match.group(0))
        if normalized is None or not math.isfinite(normalized):
            return False
    return True


def _corrupt_operator(
    step_text: str,
    *,
    rng: random.Random,
    token_counter: Callable[[str], int] | None,
    token_delta_max: int,
    retry_limit: int,
    perplexity_scorer: Callable[[str, str], float] | None,
    max_perplexity_ratio: float | None,
) -> CorruptionResult:
    replacements = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
    }
    candidates = _find_operator_matches(step_text, replacements)
    if not candidates:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
            failure_tier="uncorruptible",
        )

    rng.shuffle(candidates)
    saw_budget_failure = False
    for attempt_index, (start, end, operator, replacement) in enumerate(
        candidates[: max(retry_limit, 1)],
        start=1,
    ):
        corrupt_text = _replace_fragment(step_text, start, end, replacement)
        validation = _validate_corruption_candidate(
            clean_text=step_text,
            corrupt_text=corrupt_text,
            expected_change="operator",
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
            corrupt_text=corrupt_text,
            original_number=operator,
            perturbed_number=replacement,
            corruption_failed=False,
            corruption_tier=2,
            corruption_type="operator_swap",
            original_fragment=operator,
            corrupted_fragment=replacement,
            token_delta=token_delta,
            attempts=attempt_index,
            perplexity_ratio=perplexity_ratio,
        )

    return CorruptionResult(
        corrupt_text=step_text,
        original_number="",
        perturbed_number="",
        corruption_failed=True,
        failure_tier="tier2_token_budget" if saw_budget_failure else "uncorruptible",
    )


def _find_operator_matches(
    text: str,
    replacements: dict[str, str],
) -> list[tuple[int, int, str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    for index, char in enumerate(text):
        if char not in OPERATOR_CHARS:
            continue
        if char in {"+", "-"} and _looks_like_sign(text, index):
            continue
        if char == "-" and _is_hyphenated_word(text, index):
            continue
        replacement = replacements.get(char)
        if replacement is None:
            continue
        matches.append((index, index + 1, char, replacement))
    return matches


def _looks_like_sign(text: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    if cursor < 0:
        return True
    return text[cursor] in "(=</>"


def _is_hyphenated_word(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    return text[index - 1].isalpha() and text[index + 1].isalpha()


def _corrupt_semantic_flip(
    step_text: str,
    *,
    token_counter: Callable[[str], int] | None,
    token_delta_max: int,
    perplexity_scorer: Callable[[str, str], float] | None,
    max_perplexity_ratio: float | None,
) -> CorruptionResult:
    candidate = _find_semantic_flip_candidate(step_text)
    if candidate is None:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
            failure_tier="uncorruptible",
        )

    start, end, original_fragment, replacement = candidate
    corrupt_text = _replace_fragment(step_text, start, end, replacement)
    validation = _validate_corruption_candidate(
        clean_text=step_text,
        corrupt_text=corrupt_text,
        expected_change="semantic",
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        perplexity_scorer=perplexity_scorer,
        max_perplexity_ratio=max_perplexity_ratio,
    )
    if validation is None:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
            failure_tier="tier3_token_budget",
        )

    token_delta, perplexity_ratio = validation
    return CorruptionResult(
        corrupt_text=corrupt_text,
        original_number=original_fragment,
        perturbed_number=replacement,
        corruption_failed=False,
        corruption_tier=3,
        corruption_type="semantic_flip",
        original_fragment=original_fragment,
        corrupted_fragment=replacement,
        token_delta=token_delta,
        attempts=1,
        perplexity_ratio=perplexity_ratio,
    )


def _find_semantic_flip_candidate(text: str) -> tuple[int, int, str, str] | None:
    earliest: tuple[int, int, str, str] | None = None
    for source, target in SEMANTIC_FLIP_PAIRS:
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            continue
        replacement = _match_case(match.group(0), target)
        candidate = (match.start(), match.end(), match.group(0), replacement)
        if earliest is None or candidate[0] < earliest[0]:
            earliest = candidate
    return earliest


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def _sample_multiplier(
    *,
    float_perturbation_range: tuple[float, ...],
    rng: random.Random,
) -> float:
    _validate_float_perturbation_range(float_perturbation_range)
    if len(float_perturbation_range) == 2:
        low, high = float_perturbation_range
        return rng.uniform(low, high)

    low_a, high_a, low_b, high_b = float_perturbation_range
    width_a = high_a - low_a
    width_b = high_b - low_b
    if rng.random() < (width_a / (width_a + width_b)):
        return rng.uniform(low_a, high_a)
    return rng.uniform(low_b, high_b)


def _format_pipeline_perturbed_value(raw: str, original_value: float, multiplier: float) -> str:
    perturbed_value = original_value * multiplier
    if _is_integer_like(raw, original_value):
        rounded_up = int(math.ceil(perturbed_value))
        return _format_integer_like(raw, rounded_up)

    precision = max(_decimal_places(raw), 2)
    rounded = round(perturbed_value, precision)
    formatted = f"{rounded:.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _is_integer_like(raw: str, value: float) -> bool:
    return float(value).is_integer() and "." not in raw


def _format_integer_like(raw: str, value: int) -> str:
    if "," in raw:
        return f"{value:,}"
    return str(value)


def _decimal_places(raw: str) -> int:
    normalized = raw.replace(",", "")
    if "." not in normalized:
        return 0
    return len(normalized.split(".", maxsplit=1)[1])


def _validate_integer_perturbation_range(values: tuple[int, int]) -> None:
    low, high = values
    if not (low < 0 < high):
        raise ValueError("integer_perturbation_range must satisfy low < 0 < high.")


def _validate_float_perturbation_range(values: tuple[float, ...]) -> None:
    if len(values) == 2:
        low, high = values
        if not (0.0 < low < high):
            raise ValueError("float_perturbation_range must satisfy 0.0 < low < high.")
        return

    if len(values) == 4:
        low_a, high_a, low_b, high_b = values
        if not (0.0 < low_a < high_a < 1.0 < low_b < high_b):
            raise ValueError(
                "float_perturbation_range must satisfy "
                "0.0 < low_a < high_a < 1.0 < low_b < high_b when two "
                "disjoint intervals are provided."
            )
        return

    raise ValueError(
        "float_perturbation_range must contain either 2 floats "
        "(single interval) or 4 floats (two disjoint intervals)."
    )
