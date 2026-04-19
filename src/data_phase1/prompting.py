"""Prompt template loading and prompt-construction helpers."""

from pathlib import Path
from typing import Any

import yaml


def resolve_prompt_templates_dir(prompts_dir: str = "prompts/") -> Path:
    """Resolve the effective prompt directory, preserving the legacy root default."""

    prompt_dir = Path(prompts_dir)
    if any(prompt_dir.glob("*.yaml")):
        return prompt_dir

    # The legacy Stage 1 entrypoints historically defaulted to `prompts/`.
    # In the current repo layout, those canonical prompts live under `prompts/first_run/`.
    if prompt_dir.name == "prompts":
        first_run_dir = prompt_dir / "first_run"
        if any(first_run_dir.glob("*.yaml")):
            return first_run_dir

    return prompt_dir


def load_prompt_template(prompt_id: str, prompts_dir: str = "prompts/") -> dict:
    """Load a YAML prompt template by id."""

    prompt_dir = resolve_prompt_templates_dir(prompts_dir)
    template_path = prompt_dir / f"{prompt_id}.yaml"
    if not template_path.exists():
        for candidate_path in sorted(prompt_dir.glob("*.yaml")):
            data = _load_prompt_template_file(candidate_path)
            if data["prompt_id"] == prompt_id:
                return data
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    return _load_prompt_template_file(template_path)


def load_prompt_templates_by_id(prompts_dir: str = "prompts/") -> tuple[Path, dict[str, dict[str, Any]]]:
    """Load every YAML prompt template in a directory and key them by prompt_id."""

    prompt_dir = resolve_prompt_templates_dir(prompts_dir)
    templates_by_id: dict[str, dict[str, Any]] = {}
    for template_path in sorted(prompt_dir.glob("*.yaml")):
        template = _load_prompt_template_file(template_path)
        prompt_id = template["prompt_id"]
        if prompt_id in templates_by_id:
            raise ValueError(
                f"Duplicate prompt_id '{prompt_id}' found in '{prompt_dir}'."
            )
        templates_by_id[prompt_id] = template
    return prompt_dir, templates_by_id


def inspect_prompt_templates(prompts_dir: str = "prompts/") -> tuple[Path, list[dict[str, str]]]:
    """Return a file-by-file inventory of prompt templates for diagnostics."""

    prompt_dir = resolve_prompt_templates_dir(prompts_dir)
    inventory: list[dict[str, str]] = []
    for template_path in sorted(prompt_dir.glob("*.yaml")):
        template = _load_prompt_template_file(template_path)
        inventory.append(
            {
                "filename": template_path.name,
                "prompt_id": template["prompt_id"],
            }
        )
    return prompt_dir, inventory


def build_generation_messages(
    question: str,
    prompt_template: dict,
) -> list[dict]:
    """Build chat messages for ICL-driven generation."""

    system_template = _require_string(prompt_template, "system")
    user_template = _require_string(prompt_template, "user_template")
    few_shot = prompt_template.get("few_shot", [])
    if not isinstance(few_shot, list):
        raise TypeError("Prompt template field 'few_shot' must be a list.")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_template,
        }
    ]

    for exemplar in few_shot:
        if not isinstance(exemplar, dict):
            raise TypeError("Each few-shot exemplar must be a mapping.")
        user_content = _require_string(exemplar, "user")
        assistant_content = _require_string(exemplar, "assistant")
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append(
        {
            "role": "user",
            "content": user_template.format(question=question),
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


def _load_prompt_template_file(template_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Prompt template must be a mapping: {template_path}")
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise TypeError(f"Prompt template field 'prompt_id' must be a non-empty string: {template_path}")
    return data


def _require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Prompt template field '{key}' must be a string.")
    return value
