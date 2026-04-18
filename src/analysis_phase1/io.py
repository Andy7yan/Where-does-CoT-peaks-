"""IO helpers for the canonical analysis sample layout."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER


@dataclass(frozen=True)
class SampleRecord:
    """Resolved sample payload joined across difficulty/bin/sample files."""

    difficulty: str
    length: int
    sample_id: str
    sample_dir: Path
    source_trace_id: str
    question_id: str
    question_text: str
    gold_answer: float | int | str
    actual_num_steps: int
    trace_tier: int
    k_values: tuple[int, ...]
    clean_steps: tuple[str, ...]
    clean_raw_completion: str
    final_answer_line: str | None
    corruptions_by_k: dict[int, dict[str, Any]]
    per_k_meta: dict[int, dict[str, Any]]


def load_analysis_samples(run_dir: str | Path) -> list[SampleRecord]:
    """Load all analysis samples from the canonical difficulty tree."""

    run_path = Path(run_dir)
    difficulty_root = run_path / "difficulty"
    if not difficulty_root.exists():
        raise FileNotFoundError(f"Missing difficulty root: {difficulty_root}")

    samples: list[SampleRecord] = []
    for difficulty in DIFFICULTY_ORDER:
        difficulty_dir = difficulty_root / difficulty
        if not difficulty_dir.exists():
            continue
        question_rows = _load_jsonl(difficulty_dir / "questions.jsonl")
        questions_by_id = {
            str(row["question_id"]): row
            for row in question_rows
        }
        bins_dir = difficulty_dir / "bins"
        if not bins_dir.exists():
            continue
        for bin_dir in sorted(
            path for path in bins_dir.iterdir()
            if path.is_dir() and path.name.startswith("bin_")
        ):
            try:
                length = int(bin_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid bin directory name: {bin_dir.name}") from exc
            selection_rows = _load_jsonl(bin_dir / "selection.jsonl")
            for selection_row in selection_rows:
                sample_id = str(selection_row["sample_id"])
                sample_dir = bin_dir / "samples" / sample_id
                if not sample_dir.exists():
                    raise FileNotFoundError(f"Missing sample directory referenced by selection.jsonl: {sample_dir}")

                meta = _load_json(sample_dir / "meta.json")
                clean = _load_json(sample_dir / "clean.json")
                question_id = str(meta["question_id"])
                question_row = questions_by_id.get(question_id)
                if question_row is None:
                    raise KeyError(
                        f"Sample {difficulty}/{bin_dir.name}/{sample_id} references missing question_id {question_id!r}."
                    )

                k_values = tuple(int(value) for value in meta.get("k_values", []))
                corruptions_by_k: dict[int, dict[str, Any]] = {}
                for k in k_values:
                    corruptions_by_k[k] = _load_json(sample_dir / f"corrupt_k{k}.json")
                per_k_meta = {
                    int(row["k"]): row
                    for row in meta.get("per_k", [])
                }
                samples.append(
                    SampleRecord(
                        difficulty=difficulty,
                        length=length,
                        sample_id=sample_id,
                        sample_dir=sample_dir,
                        source_trace_id=str(meta["source_trace_id"]),
                        question_id=question_id,
                        question_text=str(question_row["question_text"]),
                        gold_answer=question_row["gold_answer"],
                        actual_num_steps=int(meta["actual_num_steps"]),
                        trace_tier=int(meta["trace_tier"]),
                        k_values=k_values,
                        clean_steps=tuple(str(step) for step in clean.get("steps", [])),
                        clean_raw_completion=str(clean.get("raw_completion", "")),
                        final_answer_line=_normalize_optional_string(clean.get("final_answer_line")),
                        corruptions_by_k=corruptions_by_k,
                        per_k_meta=per_k_meta,
                    )
                )

    return sorted(
        samples,
        key=lambda row: (
            DIFFICULTY_ORDER.index(row.difficulty),
            row.length,
            row.sample_id,
        ),
    )


def load_analysis_traces_by_difficulty(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-difficulty trace tables used for accuracy-by-length aggregation."""

    run_path = Path(run_dir)
    difficulty_root = run_path / "difficulty"
    if not difficulty_root.exists():
        raise FileNotFoundError(f"Missing difficulty root: {difficulty_root}")

    traces_by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for difficulty in DIFFICULTY_ORDER:
        traces_path = difficulty_root / difficulty / "traces.jsonl"
        traces_by_difficulty[difficulty] = _load_jsonl(traces_path) if traces_path.exists() else []
    return traces_by_difficulty


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
