"""Reasoning-text utilities for segmentation, extraction, and corruption."""

from dataclasses import dataclass
import random
import re


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.\d+|\d[\d,]*\.?|\.\d+)")
ARITHMETIC_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
DEFAULT_ANSWER_MARKERS = ["####", "The answer is"]
PUNCTUATION_ONLY_RE = re.compile(r"^\W+$")


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
    """Structured result for an arithmetic corruption attempt."""

    corrupt_text: str
    original_number: str
    perturbed_number: str
    corruption_failed: bool


@dataclass(frozen=True)
class _NumberMatch:
    """Internal representation of a replaceable numeric span."""

    start: int
    end: int
    raw: str
    value: float


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
    perturbation_range: tuple[float, float] = (0.5, 1.5),
    exclusion_zone: tuple[float, float] = (0.9, 1.1),
    rng: random.Random | None = None,
    min_value: float = 10,
) -> CorruptionResult:
    """Replace one numeric result with a perturbed value."""

    generator = rng or random.Random()
    candidates = [
        match
        for match in _find_numeric_matches(step_text)
        if abs(match.value) >= min_value
    ]
    if not candidates:
        return CorruptionResult(
            corrupt_text=step_text,
            original_number="",
            perturbed_number="",
            corruption_failed=True,
        )

    selected = generator.choice(candidates)
    multiplier = _sample_multiplier(
        perturbation_range=perturbation_range,
        exclusion_zone=exclusion_zone,
        rng=generator,
    )
    perturbed_number = _format_perturbed_value(selected.raw, selected.value, multiplier)
    corrupt_text = (
        step_text[: selected.start]
        + perturbed_number
        + step_text[selected.end :]
    )

    return CorruptionResult(
        corrupt_text=corrupt_text,
        original_number=selected.raw,
        perturbed_number=perturbed_number,
        corruption_failed=False,
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

        tail = completion[marker_index + marker_len :]
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
            )
        )
    return matches


def _sample_multiplier(
    perturbation_range: tuple[float, float],
    exclusion_zone: tuple[float, float],
    rng: random.Random,
) -> float:
    low, high = perturbation_range
    exclude_low, exclude_high = exclusion_zone

    intervals = [
        (low, min(high, exclude_low)),
        (max(low, exclude_high), high),
    ]
    valid_intervals = [
        (interval_low, interval_high)
        for interval_low, interval_high in intervals
        if interval_high > interval_low
    ]
    if not valid_intervals:
        raise ValueError("No valid multiplier interval remains after exclusion.")

    total_width = sum(
        interval_high - interval_low
        for interval_low, interval_high in valid_intervals
    )
    draw = rng.uniform(0.0, total_width)
    cursor = 0.0
    for interval_low, interval_high in valid_intervals:
        width = interval_high - interval_low
        if draw <= cursor + width:
            return interval_low + (draw - cursor)
        cursor += width

    interval_low, interval_high = valid_intervals[-1]
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
