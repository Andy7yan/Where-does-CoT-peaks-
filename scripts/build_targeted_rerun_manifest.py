"""Build a 10-question targeted rerun manifest for short/long supplementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    PER_QUESTION_SELECTION_META_FILENAME,
    load_per_question_manifest,
    save_per_question_manifest,
)


DEFAULT_TARGET_QUESTION_CODES = (
    "0967",
    "0440",
    "1204",
    "0948",
    "0294",
    "0244",
    "0015",
    "0221",
    "0996",
    "0768",
)


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run_dir)
    source_manifest_path = source_run / PER_QUESTION_MANIFEST_FILENAME
    manifest_rows = load_per_question_manifest(source_manifest_path)
    target_ids = [_normalize_question_id(value) for value in args.question_id]
    selected_rows = build_targeted_manifest(
        manifest_rows=manifest_rows,
        target_ids=target_ids,
        samples_per_prompt=args.samples_per_prompt,
    )
    metadata = build_targeted_metadata(
        source_run=source_run,
        selected_rows=selected_rows,
        target_ids=target_ids,
        samples_per_prompt=args.samples_per_prompt,
    )
    manifest_path, meta_path = save_per_question_manifest(
        args.output_dir,
        manifest=selected_rows,
        selection_metadata=metadata,
    )
    print(f"source_manifest_path: {source_manifest_path}")
    print(f"targeted_question_count: {len(selected_rows)}")
    print(f"samples_per_prompt: {args.samples_per_prompt}")
    print(f"manifest_path: {manifest_path}")
    print(f"selection_meta_path: {meta_path}")


def build_targeted_manifest(
    *,
    manifest_rows: list[dict[str, Any]],
    target_ids: list[str],
    samples_per_prompt: int,
) -> list[dict[str, Any]]:
    rows_by_id = {
        str(row["question_id"]): dict(row)
        for row in manifest_rows
    }
    missing = [question_id for question_id in target_ids if question_id not in rows_by_id]
    if missing:
        raise KeyError(f"Missing targeted question ids in source manifest: {', '.join(missing)}")

    selected_rows: list[dict[str, Any]] = []
    for question_id in target_ids:
        row = dict(rows_by_id[question_id])
        row["target_samples_per_prompt"] = int(samples_per_prompt)
        row["target_total_traces"] = int(samples_per_prompt) * 4
        selected_rows.append(row)
    return selected_rows


def build_targeted_metadata(
    *,
    source_run: Path,
    selected_rows: list[dict[str, Any]],
    target_ids: list[str],
    samples_per_prompt: int,
) -> dict[str, Any]:
    return {
        "schema_version": "per_question_selection_v1",
        "pipeline_variant": "targeted_rerun_short_long_relaxed",
        "source_run_dir": str(source_run).replace("\\", "/"),
        "source_run_name": source_run.name,
        "source_question_metadata_path": str(source_run / "question_metadata.jsonl").replace("\\", "/"),
        "selected_question_count": len(selected_rows),
        "target_question_ids": target_ids,
        "per_question_trace_policy": {
            "targeted": {
                "target_total_traces": int(samples_per_prompt) * 4,
                "target_samples_per_prompt": int(samples_per_prompt),
                "prompt_ids": ["icl_short", "icl_medium", "icl_detailed", "icl_verbose"],
            }
        },
    }


def _normalize_question_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("gsm8k_platinum_"):
        return text
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        raise ValueError(f"Could not normalize question id: {value!r}")
    return f"gsm8k_platinum_{int(digits):04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        required=True,
        help="Existing per-question run containing per_question_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Targeted rerun output directory receiving the 10-question manifest.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=list(DEFAULT_TARGET_QUESTION_CODES),
        help="Question id/code to include. Defaults to the 10 targeted questions.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=100,
        help="Supplemental samples for each of the four ICL prompts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
