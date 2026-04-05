"""Prompt template loading and prompt-construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt_template(prompt_id: str, prompts_dir: str = "prompts/") -> dict:
    """Load a YAML prompt template by id."""

    template_path = Path(prompts_dir) / f"{prompt_id}.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Prompt template must be a mapping: {template_path}")
    return data


def build_generation_messages(
    question: str,
    target_length: int,
    prompt_template: dict,
) -> list[dict]:
    """Build chat messages for length-guided generation."""

    system_template = _require_string(prompt_template, "system")
    user_template = _require_string(prompt_template, "user_template")
    few_shot = prompt_template.get("few_shot", [])
    if not isinstance(few_shot, list):
        raise TypeError("Prompt template field 'few_shot' must be a list.")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_template.format(target_length=target_length),
        }
    ]

    for exemplar in few_shot:
        if not isinstance(exemplar, dict):
            raise TypeError("Each few-shot exemplar must be a mapping.")
        user_content = _require_string(exemplar, "user").format(
            question=question,
            target_length=target_length,
        )
        assistant_content = _require_string(exemplar, "assistant").format(
            question=question,
            target_length=target_length,
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append(
        {
            "role": "user",
            "content": user_template.format(question=question, target_length=target_length),
        }
    )
    return messages


def build_nldd_clean_prompt(
    question: str,
    steps: list[str],
    answer_suffix: str = "####",
) -> str:
    """Build the NLDD clean-condition prompt text."""

    steps_block = _format_steps_block(steps)
    return f"{question}\n\n{steps_block}\n\n{answer_suffix} "


def build_nldd_corrupt_prompt(
    question: str,
    clean_steps: list[str],
    corrupt_step: str,
    corrupt_index: int,
    answer_suffix: str = "####",
) -> str:
    """Build the NLDD corrupt-condition prompt text."""

    if corrupt_index < 0 or corrupt_index >= len(clean_steps):
        raise IndexError("corrupt_index must reference an existing clean step.")

    prompt_steps = clean_steps[:corrupt_index] + [corrupt_step]
    steps_block = _format_steps_block(prompt_steps)
    return f"{question}\n\n{steps_block}\n\n{answer_suffix} "


def _format_steps_block(steps: list[str]) -> str:
    return "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, start=1))


def _require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Prompt template field '{key}' must be a string.")
    return value
