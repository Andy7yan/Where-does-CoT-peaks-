"""Shared prompt helpers for paper-faithful NLDD corruption scoring."""

from __future__ import annotations

from typing import Any, Sequence

from src.data_phase1.prompting import build_nldd_corrupt_prompt


def extract_corrupt_step_text(
    *,
    corruption_payload: dict[str, Any],
    corruption_step_index: int,
) -> str:
    """Recover the corrupted step text from either new or legacy payloads."""

    explicit_step = corruption_payload.get("corrupt_step")
    if explicit_step is not None:
        text = str(explicit_step)
        if text:
            return text

    steps = corruption_payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise KeyError(
            "Corruption payload is missing both 'corrupt_step' and a usable 'steps' list."
        )

    step_offset = corruption_step_index - 1
    if 0 <= step_offset < len(steps):
        return str(steps[step_offset])

    raise IndexError(
        f"Corruption payload steps do not cover step {corruption_step_index}: got {len(steps)} steps."
    )


def build_canonical_corrupt_prompt(
    *,
    question: str,
    clean_steps: Sequence[str],
    corruption_step_index: int,
    corruption_payload: dict[str, Any] | None = None,
    corrupt_step: str | None = None,
) -> str:
    """Build the canonical truncate-after-corruption prompt for one step index."""

    if corrupt_step is None:
        if corruption_payload is None:
            raise ValueError("Either corruption_payload or corrupt_step must be provided.")
        corrupt_step = extract_corrupt_step_text(
            corruption_payload=corruption_payload,
            corruption_step_index=corruption_step_index,
        )

    return build_nldd_corrupt_prompt(
        question=question,
        clean_steps=[str(step) for step in clean_steps],
        corrupt_step=str(corrupt_step),
        corrupt_index=corruption_step_index - 1,
    )


__all__ = [
    "build_canonical_corrupt_prompt",
    "extract_corrupt_step_text",
]
