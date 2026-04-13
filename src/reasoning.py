"""Reasoning-text utilities for segmentation, extraction, and corruption."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Callable


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.\d+|\d[\d,]*\.?|\.\d+)")
ARITHMETIC_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
DEFAULT_ANSWER_MARKERS = ["####", "The answer is"]
DEFAULT_FLOAT_PERTURBATION_RANGE = (0.5, 0.9, 1.1, 1.5)
PUNCTUATION_ONLY_RE = re.compile(r"^\W+$")
OPERATOR_CHARS = {"+", "-", "*", "/", "×", "÷"}
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
class SegmentationResult:
    """Structured result of completion step segmentation."""

    steps: list[str]
    final_answer_line: str | None
    num_steps: int


@dataclass(frozen=True)
class ExtractionResult:
    """Structured output for numeric answer extraction."""

    value: float | None
    raw_match: str | None
    extraction_failed: bool


@dataclass(frozen=True)
class CorruptionResult:
    """Structured result for a step-level corruption attempt."""

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
    match_role: str


def segment_steps(
    completion: str,
    answer_markers: list[str] | None = None,
) -> SegmentationResult:
    """Split a completion into reasoning steps using the Stage 1 rules."""

    markers = DEFAULT_ANSWER_MARKERS if answer_markers is None else answer_markers
    steps: list[str] = []
    final_answer_line: str | None = None

    for raw_segment in completion.split("\n"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if PUNCTUATION_ONLY_RE.fullmatch(segment):
            continue

        if any(marker in segment for marker in markers):
            if final_answer_line is None:
                final_answer_line = segment
            continue

        steps.append(segment)

    return SegmentationResult(
        steps=steps,
        final_answer_line=final_answer_line,
        num_steps=len(steps),
    )


def extract_answer(completion: str) -> ExtractionResult:
    """Extract a numeric answer following Stage 1 marker rules."""

    raw_match = _extract_after_marker(completion, "####")
    if raw_match is None:
        raw_match = _extract_after_marker(completion, "The answer is")

    if raw_match is None:
        return ExtractionResult(value=None, raw_match=None, extraction_failed=True)

    value = normalize_numeric(raw_match)
    return ExtractionResult(
        value=value,
        raw_match=raw_match,
        extraction_failed=value is None,
    )


def normalize_numeric(raw: str) -> float | None:
    """Normalize a numeric string by dropping formatting symbols."""

    cleaned = (
        raw.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace(" ", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def judge(extracted: float | None, gold: float, tolerance: float = 1e-3) -> bool:
    """Check whether an extracted numeric answer matches the gold value."""

    if extracted is None:
        return False
    return abs(extracted - gold) < tolerance


def corrupt_arithmetic(
    step_text: str,
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    rng: random.Random | None = None,
    min_value: float = 0.0,
    token_counter: Callable[[str], int] | None = None,
    token_delta_max: int | None = None,
    retry_limit: int = 3,
    perplexity_scorer: Callable[[str, str], float] | None = None,
    max_perplexity_ratio: float | None = None,
) -> CorruptionResult:
    """Apply Tier 1 numeric corruption with token-budget-aware retries."""

    generator = rng or random.Random()
    candidates = [
        match for match in _find_numeric_matches(step_text) if abs(match.value) >= min_value
    ]
    if not candidates:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
            failure_tier="uncorruptible",
        )

    prioritized = _prioritize_numeric_candidates(candidates, rng=generator)
    saw_budget_failure = False

    for selected in prioritized:
        if _is_integer_like(selected.raw, selected.value):
            deltas = _valid_integer_deltas(selected.value)
            generator.shuffle(deltas)
            for attempt_index, delta in enumerate(deltas[: max(retry_limit, 1)], start=1):
                perturbed_value = int(round(selected.value)) + delta
                perturbed_number = _format_integer_like(selected.raw, perturbed_value)
                corrupt_text = _replace_fragment(
                    step_text,
                    selected.start,
                    selected.end,
                    perturbed_number,
                )
                validation = _validate_corruption_candidate(
                    clean_text=step_text,
                    corrupt_text=corrupt_text,
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
                    corruption_type=_numeric_corruption_type(selected),
                    original_fragment=selected.raw,
                    corrupted_fragment=perturbed_number,
                    token_delta=token_delta,
                    attempts=attempt_index,
                    perplexity_ratio=perplexity_ratio,
                )
            continue

        for attempt_index in range(1, max(retry_limit, 1) + 1):
            multiplier = _sample_multiplier(
                float_perturbation_range=float_perturbation_range,
                rng=generator,
            )
            perturbed_number = _format_perturbed_value(
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
                corruption_type=_numeric_corruption_type(selected),
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
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
    token_counter: Callable[[str], int] | None = None,
    token_delta_max: int = 2,
    retry_limit: int = 3,
    use_tier3: bool | None = None,
    perplexity_scorer: Callable[[str, str], float] | None = None,
    max_perplexity_ratio: float | None = None,
) -> CorruptionResult:
    """Apply the layered corruption strategy with tiered fallback."""

    generator = rng or random.Random()
    tier3_enabled = enable_tier3_semantic_flip or bool(use_tier3)
    tier1 = corrupt_arithmetic(
        step_text,
        float_perturbation_range=float_perturbation_range,
        rng=generator,
        min_value=0.0,
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        perplexity_scorer=perplexity_scorer,
        max_perplexity_ratio=max_perplexity_ratio,
    )
    if not tier1.corruption_failed:
        return tier1

    tier2 = _corrupt_operator(
        step_text,
        rng=generator,
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        perplexity_scorer=perplexity_scorer,
        max_perplexity_ratio=max_perplexity_ratio,
    )
    if not tier2.corruption_failed:
        return tier2

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

    failure_tier = tier1.failure_tier or tier2.failure_tier or tier3_failure_tier or "uncorruptible"
    return CorruptionResult(
        corrupt_text=step_text,
        original_number="",
        perturbed_number="",
        corruption_failed=True,
        failure_tier=failure_tier,
    )


def corrupt_step_text(step_text: str) -> str | None:
    """Backward-compatible wrapper around arithmetic corruption."""

    result = corrupt_arithmetic(step_text)
    if result.corruption_failed:
        return None
    return result.corrupt_text


def _extract_after_marker(completion: str, marker: str) -> str | None:
    marker_len = len(marker)
    start = 0
    while True:
        marker_index = completion.find(marker, start)
        if marker_index == -1:
            return None

        tail = completion[marker_index + marker_len:]
        match = NUMBER_RE.search(tail)
        if match is not None:
            return match.group(0)

        start = marker_index + marker_len


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
                match_role="result" if _is_rhs_of_equals(text, match.start()) else "operand",
            )
        )
    return matches


def _prioritize_numeric_candidates(
    candidates: list[_NumberMatch],
    *,
    rng: random.Random,
) -> list[_NumberMatch]:
    results = [candidate for candidate in candidates if candidate.match_role == "result"]
    operands = [candidate for candidate in candidates if candidate.match_role != "result"]
    rng.shuffle(results)
    rng.shuffle(operands)
    return [*results, *operands]


def _is_rhs_of_equals(text: str, start_index: int) -> bool:
    cursor = start_index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    if cursor >= 0 and text[cursor] == "=":
        return True

    prefix = text[max(0, start_index - 4):start_index]
    if prefix.endswith(">>"):
        return text.rfind("=", 0, start_index) != -1 and text.rfind("<<", 0, start_index) != -1
    return False


def _numeric_corruption_type(match: _NumberMatch) -> str:
    if match.match_role == "result":
        return "numeric_result"
    return "numeric_operand"


def _is_integer_like(raw: str, value: float) -> bool:
    return float(value).is_integer() and "." not in raw


def _valid_integer_deltas(value: float) -> list[int]:
    rounded = int(round(value))
    if rounded == 0:
        return [1, 2]
    return [delta for delta in (-2, -1, 1, 2) if rounded + delta >= 0]


def _format_integer_like(raw: str, value: int) -> str:
    if "," in raw:
        return f"{value:,}"
    return str(value)


def _replace_fragment(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _validate_corruption_candidate(
    *,
    clean_text: str,
    corrupt_text: str,
    token_counter: Callable[[str], int] | None,
    token_delta_max: int | None,
    perplexity_scorer: Callable[[str, str], float] | None,
    max_perplexity_ratio: float | None,
) -> tuple[int, float | None] | None:
    if clean_text == corrupt_text:
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
        "*": "+",
        "×": "+",
        "/": "*",
        "÷": "×",
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
    float_perturbation_range: tuple[float, float, float, float],
    rng: random.Random,
) -> float:
    low_min, low_max, high_min, high_max = float_perturbation_range
    if not (low_min < low_max < high_min < high_max):
        raise ValueError(
            "float_perturbation_range must satisfy "
            "low_min < low_max < high_min < high_max."
        )

    interval_low, interval_high = (
        (low_min, low_max) if rng.random() < 0.5 else (high_min, high_max)
    )
    return rng.uniform(interval_low, interval_high)


def _format_perturbed_value(raw: str, original_value: float, multiplier: float) -> str:
    precision = _decimal_places(raw)
    perturbed_value = original_value * multiplier

    if precision == 0:
        return str(int(round(perturbed_value)))

    rounded = round(perturbed_value, precision)
    return f"{rounded:.{precision}f}"


def _decimal_places(raw: str) -> int:
    normalized = raw.replace(",", "")
    if "." not in normalized:
        return 0
    return len(normalized.split(".", maxsplit=1)[1])
