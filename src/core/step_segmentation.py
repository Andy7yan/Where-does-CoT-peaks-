"""Utilities for segmenting completions into reasoning steps."""

from dataclasses import dataclass
import re


DEFAULT_ANSWER_MARKERS = ["####", "The answer is"]
PUNCTUATION_ONLY_RE = re.compile(r"^\W+$")


@dataclass(frozen=True)
class SegmentationResult:
    """Structured result of completion step segmentation."""

    steps: list[str]
    final_answer_line: str | None
    num_steps: int


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
