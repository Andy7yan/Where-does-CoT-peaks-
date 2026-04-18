"""Shared constants and helpers for the NLDD analysis workflow."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


TRACE_SELECTION_REQUIRED_COLUMNS = (
    "trace_id",
    "question_id",
    "difficulty",
    "length_bin",
    "selected_for_nldd",
    "selected_for_near_lstar",
)
TRACE_SELECTION_FIELDNAMES = (
    "trace_id",
    "question_id",
    "difficulty",
    "length_bin",
    "raw_length_bin",
    "actual_clean_length",
    "prompt_id",
    "selected_for_nldd",
    "selected_for_near_lstar",
    "selection_mode",
    "near_lstar_selection_mode",
)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_gold_answer_variants(gold_answer: float | int | str) -> list[str]:
    if isinstance(gold_answer, str):
        base = gold_answer.strip()
    elif isinstance(gold_answer, int):
        base = str(gold_answer)
    else:
        numeric = float(gold_answer)
        if numeric.is_integer():
            base = str(int(numeric))
        else:
            base = format(numeric, "g")
    variants = [base]
    if not base.startswith(" "):
        variants.append(f" {base}")
    return variants


def _flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            flattened: list[int] = []
            for item in value:
                flattened.extend(_flatten_token_ids(item))
            return flattened
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return [int(value)]


def _flatten_numeric_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return [float(value)]


def _compute_vector_std(logits: Any) -> float:
    values = _flatten_numeric_values(logits)
    if not values:
        raise ValueError("Cannot compute a standard deviation from an empty logit vector.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _move_model_inputs_to_device(model_inputs: Any, device: Any) -> dict[str, Any]:
    if isinstance(model_inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
    raise TypeError("Tokenizer returned an unsupported input type for NLDD scoring.")


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Could not parse a boolean value from {value!r}.")
