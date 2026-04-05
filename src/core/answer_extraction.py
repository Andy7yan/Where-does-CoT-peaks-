"""Answer extraction helpers for GSM8K-style completions."""

from __future__ import annotations

from dataclasses import dataclass
import re


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.\d+|\d[\d,]*\.?|\.\d+)")
MARKER_PATTERNS = ("####", "The answer is")


@dataclass(frozen=True)
class ExtractionResult:
    """Structured output for numeric answer extraction."""

    value: float | None
    raw_match: str | None
    extraction_failed: bool


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
