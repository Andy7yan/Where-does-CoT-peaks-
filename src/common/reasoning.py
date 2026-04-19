"""Reasoning-text utilities for segmentation and answer extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.\d+|\d[\d,]*\.?|\.\d+)")
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


from src.common.corruption import (  # noqa: E402
    corrupt_arithmetic,
    corrupt_step_text,
    corrupt_step_text_with_fallbacks,
)


__all__ = [
    "DEFAULT_ANSWER_MARKERS",
    "ExtractionResult",
    "SegmentationResult",
    "corrupt_arithmetic",
    "corrupt_step_text",
    "corrupt_step_text_with_fallbacks",
    "extract_answer",
    "judge",
    "normalize_numeric",
    "segment_steps",
]
