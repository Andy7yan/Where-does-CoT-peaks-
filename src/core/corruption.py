"""Arithmetic corruption helpers for NLDD measurements."""

from dataclasses import dataclass
import random
import re

from src.core.answer_extraction import normalize_numeric


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


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


def _find_numeric_matches(text: str) -> list[_NumberMatch]:
    matches: list[_NumberMatch] = []
    for match in NUMBER_RE.finditer(text):
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

    total_width = sum(interval_high - interval_low for interval_low, interval_high in valid_intervals)
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

def corrupt_step_text(step_text: str) -> str | None:
    """Backward-compatible wrapper around arithmetic corruption."""

    result = corrupt_arithmetic(step_text)
    if result.corruption_failed:
        return None
    return result.corrupt_text
