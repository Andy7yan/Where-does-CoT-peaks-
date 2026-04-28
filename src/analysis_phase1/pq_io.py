"""IO helpers for the per-question sample layout."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.analysis_phase1.io import SampleRecord
from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    load_per_question_manifest,
)
from src.data_phase2.per_question_pipeline import (
    PER_QUESTION_DIRNAME,
    QUESTION_METADATA_FILENAME,
)


BIN_SUMMARY_FILENAME = "bin_summary.json"
PQ_ANALYSIS_DIRNAME = "pq_analysis"


def load_per_question_samples(run_dir: str | Path) -> list[SampleRecord]:
    """Load retained clean samples from the per-question handoff tree."""

    run_path = Path(run_dir)
    per_question_root = run_path / PER_QUESTION_DIRNAME
    if not per_question_root.exists():
        raise FileNotFoundError(f"Missing per-question root: {per_question_root}")

    manifest_rows = load_per_question_manifest(run_path / PER_QUESTION_MANIFEST_FILENAME)
    manifest_by_question = {
        str(row["question_id"]): row
        for row in manifest_rows
    }
    metadata_by_question = {
        str(row["question_id"]): row
        for row in _load_jsonl(run_path / QUESTION_METADATA_FILENAME)
    }

    samples: list[SampleRecord] = []
    for question_dir in sorted(
        path for path in per_question_root.iterdir()
        if path.is_dir()
    ):
        question_id = question_dir.name
        manifest_row = manifest_by_question.get(question_id)
        if manifest_row is None:
            raise KeyError(f"Per-question sample tree references unknown question_id '{question_id}'.")
        question_meta = metadata_by_question.get(question_id)
        if question_meta is None:
            raise KeyError(f"Missing root metadata row for question_id '{question_id}'.")
        bins_dir = question_dir / "bins"
        if not bins_dir.exists():
            continue
        for bin_dir in sorted(
            path for path in bins_dir.iterdir()
            if path.is_dir() and path.name.startswith("bin_")
        ):
            try:
                length = int(bin_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid per-question bin directory name: {bin_dir.name}") from exc
            selection_rows = _load_jsonl(bin_dir / "selection.jsonl")
            for selection_row in selection_rows:
                sample_id = str(selection_row["sample_id"])
                sample_dir = bin_dir / "samples" / sample_id
                if not sample_dir.exists():
                    raise FileNotFoundError(f"Missing sample directory referenced by selection.jsonl: {sample_dir}")
                meta = _load_json(sample_dir / "meta.json")
                clean = _load_json(sample_dir / "clean.json")
                k_values = tuple(int(value) for value in meta.get("k_values", []))
                corruptions_by_k = {
                    k: _load_json(sample_dir / f"corrupt_k{k}.json")
                    for k in k_values
                }
                per_k_meta = {
                    int(row["k"]): row
                    for row in meta.get("per_k", [])
                }
                samples.append(
                    SampleRecord(
                        difficulty=str(question_meta["difficulty"]),
                        length=length,
                        sample_id=sample_id,
                        sample_dir=sample_dir,
                        source_trace_id=str(meta["source_trace_id"]),
                        question_id=question_id,
                        task_name=str(manifest_row.get("task_name", "gsm8k")),
                        question_text=str(manifest_row["question_text"]),
                        gold_answer=manifest_row["gold_answer"],
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
        key=lambda row: (row.question_id, row.length, row.sample_id),
    )


def load_per_question_metadata(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load root per-question metadata rows."""

    return _load_jsonl(Path(run_dir) / QUESTION_METADATA_FILENAME)


def load_per_question_lstar_payloads(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load one l_star.json payload per question."""

    run_path = Path(run_dir)
    payloads: dict[str, dict[str, Any]] = {}
    per_question_root = run_path / PER_QUESTION_DIRNAME
    for question_dir in sorted(
        path for path in per_question_root.iterdir()
        if path.is_dir()
    ):
        lstar_path = question_dir / "l_star.json"
        if lstar_path.exists():
            payloads[question_dir.name] = _load_json(lstar_path)
    return payloads


def load_per_question_bin_summaries(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load all per-question bin summary payloads."""

    run_path = Path(run_dir)
    summaries: list[dict[str, Any]] = []
    per_question_root = run_path / PER_QUESTION_DIRNAME
    if not per_question_root.exists():
        raise FileNotFoundError(f"Missing per-question root: {per_question_root}")
    for question_dir in sorted(
        path for path in per_question_root.iterdir()
        if path.is_dir()
    ):
        bins_dir = question_dir / "bins"
        if not bins_dir.exists():
            continue
        for bin_dir in sorted(
            path for path in bins_dir.iterdir()
            if path.is_dir() and path.name.startswith("bin_")
        ):
            summary_path = bin_dir / BIN_SUMMARY_FILENAME
            if summary_path.exists():
                summaries.append(_load_json(summary_path))
    return summaries


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


__all__ = [
    "BIN_SUMMARY_FILENAME",
    "PQ_ANALYSIS_DIRNAME",
    "load_per_question_bin_summaries",
    "load_per_question_lstar_payloads",
    "load_per_question_metadata",
    "load_per_question_samples",
]
