"""Merge old and supplemental targeted traces into a fresh relaxed PQ run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_targeted_rerun_manifest import DEFAULT_TARGET_QUESTION_CODES
from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    PER_QUESTION_SELECTION_META_FILENAME,
)


def main() -> None:
    args = parse_args()
    target_ids = {_normalize_question_id(value) for value in args.question_id}
    source_run = Path(args.source_run_dir)
    supplemental_run = Path(args.supplemental_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_traces = [
        row for row in load_traces(source_run)
        if str(row.get("question_id")) in target_ids
    ]
    supplemental_traces = [
        _rename_supplemental_trace(row, suffix=args.supplemental_suffix)
        for row in load_traces(supplemental_run)
        if str(row.get("question_id")) in target_ids
    ]
    merged_traces = merge_without_duplicate_trace_ids(
        [*source_traces, *supplemental_traces]
    )

    copy_manifest_files(
        supplemental_run=supplemental_run,
        source_run=source_run,
        output_dir=output_dir,
    )
    write_jsonl(output_dir / "traces.jsonl", merged_traces)
    write_run_meta(
        output_dir / "run_meta.json",
        source_run=source_run,
        supplemental_run=supplemental_run,
        source_trace_count=len(source_traces),
        supplemental_trace_count=len(supplemental_traces),
        merged_trace_count=len(merged_traces),
        target_ids=sorted(target_ids),
    )
    print(f"source_trace_count: {len(source_traces)}")
    print(f"supplemental_trace_count: {len(supplemental_traces)}")
    print(f"merged_trace_count: {len(merged_traces)}")
    print(f"output_dir: {output_dir}")
    print(f"traces_path: {output_dir / 'traces.jsonl'}")


def load_traces(run_dir: Path) -> list[dict[str, Any]]:
    root_traces = run_dir / "traces.jsonl"
    if root_traces.exists():
        return read_jsonl(root_traces)

    shard_root = run_dir / "shards"
    if shard_root.exists():
        rows: list[dict[str, Any]] = []
        for shard_path in sorted(shard_root.glob("*/traces.jsonl")):
            rows.extend(read_jsonl(shard_path))
        if rows:
            return rows

    raise FileNotFoundError(
        f"No traces found under {run_dir}; expected traces.jsonl or shards/*/traces.jsonl."
    )


def _rename_supplemental_trace(row: dict[str, Any], *, suffix: str) -> dict[str, Any]:
    renamed = dict(row)
    renamed["trace_id"] = f"{row['trace_id']}_{suffix}"
    renamed["source_trace_id_before_targeted_suffix"] = str(row["trace_id"])
    renamed["targeted_rerun_suffix"] = suffix
    return renamed


def merge_without_duplicate_trace_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    duplicates: list[str] = []
    for row in rows:
        trace_id = str(row.get("trace_id", ""))
        if trace_id in seen:
            duplicates.append(trace_id)
            continue
        seen.add(trace_id)
        merged.append(row)
    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate trace ids after targeted merge: {sample}")
    return merged


def copy_manifest_files(*, supplemental_run: Path, source_run: Path, output_dir: Path) -> None:
    for filename in (PER_QUESTION_MANIFEST_FILENAME, PER_QUESTION_SELECTION_META_FILENAME):
        source_path = supplemental_run / filename
        if not source_path.exists():
            source_path = source_run / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required manifest file: {filename}")
        shutil.copyfile(source_path, output_dir / filename)


def write_run_meta(
    path: Path,
    *,
    source_run: Path,
    supplemental_run: Path,
    source_trace_count: int,
    supplemental_trace_count: int,
    merged_trace_count: int,
    target_ids: list[str],
) -> None:
    payload = {
        "schema_version": "stage1_trace_v2",
        "pipeline_variant": "targeted_rerun_short_long_relaxed",
        "source_run_dir": str(source_run).replace("\\", "/"),
        "supplemental_run_dir": str(supplemental_run).replace("\\", "/"),
        "source_trace_count": source_trace_count,
        "supplemental_trace_count": supplemental_trace_count,
        "merged_trace_count": merged_trace_count,
        "target_question_ids": target_ids,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument("--source-run-dir", required=True, help="Original PQ run.")
    parser.add_argument("--supplemental-run-dir", required=True, help="Targeted generation run.")
    parser.add_argument("--output-dir", required=True, help="Fresh merged run directory.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=list(DEFAULT_TARGET_QUESTION_CODES),
        help="Target question id/code. Defaults to the 10 selected questions.",
    )
    parser.add_argument(
        "--supplemental-suffix",
        default="targeted_relaxed",
        help="Unique suffix appended to supplemental trace ids.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
