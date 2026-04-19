"""Question-selection helpers for the per-question generation path."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from src.common.settings import ExperimentConfig
from src.data_phase1.gsm8k import build_ranked_questions, load_gsm8k_test


PER_QUESTION_PIPELINE_VARIANT = "per_question"
PER_QUESTION_SOURCE_DIFFICULTIES = ("medium", "hard")
PER_QUESTION_MANIFEST_FILENAME = "per_question_manifest.jsonl"
PER_QUESTION_SELECTION_META_FILENAME = "per_question_selection_meta.json"
PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD = 7200
PER_QUESTION_TARGET_TOTAL_TRACES = {
    "medium": 120,
    "hard": 300,
}
PER_QUESTION_TARGET_SAMPLES_PER_PROMPT = {
    "medium": 30,
    "hard": 75,
}


def build_per_question_manifest(
    *,
    config_path: str,
    source_run: str | Path,
    source: str = "huggingface",
    local_path: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Build the deterministic per-question manifest from a frozen source run."""

    config = ExperimentConfig.from_yaml(config_path)
    source_run_path = resolve_source_run_dir(source_run)
    source_metadata_path = source_run_path / "question_metadata.jsonl"
    metadata_rows = _load_jsonl(source_metadata_path)
    if not metadata_rows:
        raise ValueError(f"No question metadata rows found at '{source_metadata_path}'.")

    selected_by_question_id: dict[str, str] = {}
    for row in metadata_rows:
        question_id = str(row["question_id"])
        difficulty_bucket = str(row.get("difficulty_bucket", ""))
        if difficulty_bucket not in PER_QUESTION_SOURCE_DIFFICULTIES:
            continue
        selected_by_question_id[question_id] = difficulty_bucket

    if not selected_by_question_id:
        raise ValueError(
            "The source metadata did not contain any medium/hard questions for per-question selection."
        )

    ranked_questions = build_ranked_questions(
        load_gsm8k_test(
            source=source,
            local_path=local_path,
            cache_dir=cache_dir,
            dataset_name=config.dataset.name,
            dataset_config=config.dataset.hf_config,
            split=config.dataset.split,
        ),
        hash_seed=config.dataset.order_hash_seed,
        dataset_name=config.dataset.name,
        split=config.dataset.split,
    )

    manifest: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for record in ranked_questions:
        question_id = str(record["question_id"])
        difficulty_bucket = selected_by_question_id.get(question_id)
        if difficulty_bucket is None:
            continue
        seen_question_ids.add(question_id)
        manifest.append(
            {
                "question_id": question_id,
                "question_text": str(record["question_text"]),
                "gold_answer": record["gold_answer"],
                "source_difficulty_bucket": difficulty_bucket,
                "target_total_traces": PER_QUESTION_TARGET_TOTAL_TRACES[difficulty_bucket],
                "target_samples_per_prompt": PER_QUESTION_TARGET_SAMPLES_PER_PROMPT[
                    difficulty_bucket
                ],
            }
        )

    missing_question_ids = sorted(set(selected_by_question_id) - seen_question_ids)
    if missing_question_ids:
        sample = ", ".join(missing_question_ids[:5])
        raise KeyError(
            "Failed to recover all per-question-selected questions from the ranked dataset. "
            f"Examples: {sample}"
        )

    return manifest


def build_per_question_selection_metadata(
    *,
    config_path: str,
    source_run: str | Path,
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the run-root provenance payload for per-question selection."""

    config = ExperimentConfig.from_yaml(config_path)
    source_run_path = resolve_source_run_dir(source_run)
    difficulty_counts = Counter(str(row["source_difficulty_bucket"]) for row in manifest)
    return {
        "schema_version": "per_question_selection_v1",
        "pipeline_variant": PER_QUESTION_PIPELINE_VARIANT,
        "source_run_dir": str(source_run_path).replace("\\", "/"),
        "source_run_name": source_run_path.name,
        "source_question_metadata_path": str(
            source_run_path / "question_metadata.jsonl"
        ).replace("\\", "/"),
        "dataset": f"{config.dataset.name}:{config.dataset.split}",
        "selected_question_count": len(manifest),
        "difficulty_counts": {
            difficulty: int(difficulty_counts.get(difficulty, 0))
            for difficulty in PER_QUESTION_SOURCE_DIFFICULTIES
        },
        "per_question_trace_policy": {
            difficulty: {
                "target_total_traces": PER_QUESTION_TARGET_TOTAL_TRACES[difficulty],
                "target_samples_per_prompt": PER_QUESTION_TARGET_SAMPLES_PER_PROMPT[difficulty],
            }
            for difficulty in PER_QUESTION_SOURCE_DIFFICULTIES
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_per_question_manifest(
    output_dir: str | Path,
    *,
    manifest: list[dict[str, Any]],
    selection_metadata: dict[str, Any],
    manifest_filename: str = PER_QUESTION_MANIFEST_FILENAME,
    meta_filename: str = PER_QUESTION_SELECTION_META_FILENAME,
) -> tuple[str, str]:
    """Write the per-question manifest and its provenance payload."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / manifest_filename
    selection_meta_path = output_path / meta_filename

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selection_meta_path.write_text(
        json.dumps(selection_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(manifest_path), str(selection_meta_path)


def load_per_question_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a per-question manifest from disk."""

    manifest_path = Path(path)
    rows = _load_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"No per-question manifest rows found at '{manifest_path}'.")
    required_fields = {
        "question_id",
        "question_text",
        "gold_answer",
        "source_difficulty_bucket",
        "target_total_traces",
        "target_samples_per_prompt",
    }
    for index, row in enumerate(rows, start=1):
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            raise ValueError(
                "Per-question manifest row "
                f"{index} is missing required fields: {', '.join(missing_fields)}"
            )
    return rows


def slice_per_question_manifest(
    manifest_rows: list[dict[str, Any]],
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[dict[str, Any]]:
    """Take a deterministic slice from a loaded per-question manifest."""

    total_questions = len(manifest_rows)
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative.")
    if start_idx > total_questions:
        raise ValueError(
            f"start_idx {start_idx} exceeds the per-question manifest size {total_questions}."
        )
    effective_end_idx = total_questions if end_idx is None else end_idx
    if effective_end_idx < start_idx:
        raise ValueError("end_idx must be greater than or equal to start_idx.")
    if effective_end_idx > total_questions:
        raise ValueError(
            f"end_idx {effective_end_idx} exceeds the per-question manifest size {total_questions}."
        )
    return list(manifest_rows[start_idx:effective_end_idx])


def load_per_question_selection_metadata(path: str | Path) -> dict[str, Any]:
    """Load per-question selection metadata when present."""

    selection_meta_path = Path(path)
    if not selection_meta_path.exists():
        raise FileNotFoundError(
            f"Per-question selection metadata not found: {selection_meta_path}"
        )
    return json.loads(selection_meta_path.read_text(encoding="utf-8"))


def infer_per_question_shard_count(
    manifest_rows: list[dict[str, Any]],
    *,
    questions_per_shard: int | None = None,
    target_traces_per_shard: int = PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD,
) -> int:
    """Infer a shard count from explicit question budgets or target trace load."""

    total_questions = len(manifest_rows)
    if total_questions <= 0:
        raise ValueError("Per-question shard planning requires at least one manifest row.")

    if questions_per_shard is not None:
        if questions_per_shard <= 0:
            raise ValueError("questions_per_shard must be positive when provided.")
        return max(1, (total_questions + questions_per_shard - 1) // questions_per_shard)

    if target_traces_per_shard <= 0:
        raise ValueError("target_traces_per_shard must be positive.")

    total_target_traces = sum(_coerce_target_traces(row) for row in manifest_rows)
    return max(1, (total_target_traces + target_traces_per_shard - 1) // target_traces_per_shard)


def plan_per_question_shards(
    manifest_rows: list[dict[str, Any]],
    *,
    shard_count: int,
) -> list[dict[str, int]]:
    """Plan contiguous shards with approximately equal target-trace load."""

    total_questions = len(manifest_rows)
    if total_questions <= 0:
        raise ValueError("Per-question shard planning requires at least one manifest row.")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if shard_count > total_questions:
        raise ValueError("shard_count cannot exceed the number of manifest rows.")

    weights = [_coerce_target_traces(row) for row in manifest_rows]
    total_remaining_traces = sum(weights)
    start_idx = 0
    plans: list[dict[str, int]] = []

    for shard_index in range(shard_count):
        remaining_shards = shard_count - shard_index
        max_end_idx = total_questions - (remaining_shards - 1)
        current_weight = 0
        best_end_idx = start_idx + 1
        best_diff = float("inf")
        target_traces = total_remaining_traces / remaining_shards

        for candidate_end_idx in range(start_idx + 1, max_end_idx + 1):
            current_weight += weights[candidate_end_idx - 1]
            diff = abs(current_weight - target_traces)
            if diff < best_diff or (
                diff == best_diff and candidate_end_idx > best_end_idx
            ):
                best_diff = diff
                best_end_idx = candidate_end_idx

        shard_traces = sum(weights[start_idx:best_end_idx])
        plans.append(
            {
                "shard_index": shard_index,
                "start_idx": start_idx,
                "end_idx": best_end_idx,
                "question_count": best_end_idx - start_idx,
                "target_total_traces": shard_traces,
            }
        )
        start_idx = best_end_idx
        total_remaining_traces -= shard_traces

    return plans


def resolve_source_run_dir(source_run: str | Path) -> Path:
    """Resolve a source run path, preferring the Katana scratch run layout."""

    raw_value = str(source_run).strip()
    if not raw_value:
        raise ValueError("source_run must be a non-empty path or run name.")

    candidate = Path(raw_value)
    candidates: list[Path] = []

    scratch_root = os.getenv("SCRATCH")
    if scratch_root and not candidate.is_absolute():
        scratch_runs_root = Path(scratch_root) / "runs"
        normalized_parts = list(candidate.parts)
        if normalized_parts[:1] == ["results"] and len(normalized_parts) >= 2:
            candidates.append(scratch_runs_root / Path(*normalized_parts[1:]))
        candidates.append(scratch_runs_root / candidate)
        candidates.append(scratch_runs_root / candidate.name)

    # Exact path fallback after scratch candidates, because Katana run outputs live under
    # `${SCRATCH}/runs/...` and we want that contract to win over stale local `results/...`.
    candidates.append(candidate)

    for path in _dedupe_paths(candidates):
        if (path / "question_metadata.jsonl").exists():
            return path

    formatted_candidates = "\n".join(
        f"- {str(path).replace(chr(92), '/')}" for path in _dedupe_paths(candidates)
    )
    raise FileNotFoundError(
        "Could not resolve the source run directory. Checked these candidates:\n"
        f"{formatted_candidates}"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _coerce_target_traces(row: dict[str, Any]) -> int:
    value = row.get("target_total_traces")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Per-question manifest rows must contain integer target_total_traces.")
    return value


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


__all__ = [
    "PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD",
    "PER_QUESTION_MANIFEST_FILENAME",
    "PER_QUESTION_PIPELINE_VARIANT",
    "PER_QUESTION_SELECTION_META_FILENAME",
    "PER_QUESTION_SOURCE_DIFFICULTIES",
    "PER_QUESTION_TARGET_SAMPLES_PER_PROMPT",
    "PER_QUESTION_TARGET_TOTAL_TRACES",
    "build_per_question_manifest",
    "build_per_question_selection_metadata",
    "infer_per_question_shard_count",
    "load_per_question_manifest",
    "load_per_question_selection_metadata",
    "plan_per_question_shards",
    "resolve_source_run_dir",
    "save_per_question_manifest",
    "slice_per_question_manifest",
]
