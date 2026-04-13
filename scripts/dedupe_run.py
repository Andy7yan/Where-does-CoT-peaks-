"""Deduplicate Stage 1 traces and attached corruption artifacts in-place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nldd import summarize_corruption_records
from src.reports import (
    aggregate_stage1_outputs,
    discover_stage1_shard_paths,
    ensure_root_run_metadata,
    merge_stage1_shards,
)


def main() -> None:
    args = parse_args()
    run_path = Path(args.run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    removed_trace_ids, shard_stats = dedupe_shard_traces(run_path)
    root_stats = rewrite_root_traces(run_path)
    legacy_stats = dedupe_legacy_failed_corruptions(run_path, removed_trace_ids)
    corruption_stats = dedupe_corruption_artifacts(run_path, removed_trace_ids)
    aggregation_stats = aggregate_stage1_outputs(str(run_path), config_path=args.config)

    summary = {
        "run_dir": str(run_path),
        "removed_trace_ids": len(removed_trace_ids),
        "shards": shard_stats,
        "root_traces": root_stats,
        "failed_corruptions": legacy_stats,
        "corruptted_traces": corruption_stats,
        "aggregation": aggregation_stats,
        "stale_manual_artifacts": [
            "stage1_analysis_report.md",
            "manual_review_items.md",
        ],
    }
    summary_path = run_path / "dedupe_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"dedupe_summary: {summary_path}")
    print(f"removed_trace_ids: {len(removed_trace_ids)}")
    print(f"root_traces_before: {root_stats['before']}")
    print(f"root_traces_after: {root_stats['after']}")
    print(f"root_traces_removed: {root_stats['removed']}")
    print("note: stage1_analysis_report.md and manual_review_items.md are not regenerated automatically.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing run directory under $SCRATCH/runs.")
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Config used when regenerating accuracy_by_length.csv and question_metadata.jsonl.",
    )
    return parser.parse_args()


def dedupe_shard_traces(run_path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    shard_paths = discover_stage1_shard_paths(run_path)
    if not shard_paths:
        raise FileNotFoundError(f"No shard traces found under '{run_path / 'shards'}'.")

    removed_trace_ids: set[str] = set()
    shard_stats: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        rows = _load_jsonl(shard_path)
        deduped, removed = _dedupe_trace_rows(rows)
        removed_trace_ids.update(removed)
        _write_jsonl(shard_path, deduped)
        shard_stats.append(
            {
                "shard_id": shard_path.parent.name,
                "before": len(rows),
                "after": len(deduped),
                "removed": len(rows) - len(deduped),
            }
        )
    return removed_trace_ids, shard_stats


def rewrite_root_traces(run_path: Path) -> dict[str, int]:
    root_traces_path = run_path / "traces.jsonl"
    before = len(_load_jsonl(root_traces_path)) if root_traces_path.exists() else 0
    shard_paths = discover_stage1_shard_paths(run_path)
    merged = merge_stage1_shards(shard_paths)
    _write_jsonl(root_traces_path, merged)
    ensure_root_run_metadata(run_path, shard_paths)
    return {
        "before": before,
        "after": len(merged),
        "removed": max(before - len(merged), 0),
    }


def dedupe_legacy_failed_corruptions(
    run_path: Path,
    removed_trace_ids: set[str],
) -> dict[str, int] | None:
    path = run_path / "failed_corruptions.jsonl"
    if not path.exists():
        return None

    rows = _load_jsonl(path)
    filtered = [row for row in rows if str(row.get("trace_id", "")) not in removed_trace_ids]
    deduped = _dedupe_rows(
        filtered,
        key_fn=lambda row: (
            row.get("trace_id"),
            row.get("question_id"),
            row.get("step_index"),
            row.get("step_text"),
            row.get("actual_num_steps"),
            row.get("failure_reason"),
        ),
    )
    _write_jsonl(path, deduped)
    return {"before": len(rows), "after": len(deduped), "removed": len(rows) - len(deduped)}


def dedupe_corruption_artifacts(
    run_path: Path,
    removed_trace_ids: set[str],
) -> dict[str, Any] | None:
    corruption_dir = run_path / "corruptted_traces"
    if not corruption_dir.exists():
        return None

    mode_rows: dict[str, list[dict[str, Any]]] = {}
    mode_stats: dict[str, dict[str, int]] = {}
    for mode_name in ("all_steps", "sampled_steps"):
        path = corruption_dir / f"{mode_name}.jsonl"
        if not path.exists():
            continue
        rows = _load_jsonl(path)
        filtered = [row for row in rows if str(row.get("trace_id", "")) not in removed_trace_ids]
        deduped = _dedupe_rows(
            filtered,
            key_fn=lambda row: (
                row.get("corruption_id"),
                row.get("trace_id"),
                row.get("step_index"),
                row.get("selection_mode"),
                row.get("clean_step"),
                row.get("corrupt_step"),
                row.get("corruption_failed"),
                row.get("corruption_type"),
                row.get("failure_tier"),
            ),
        )
        _write_jsonl(path, deduped)
        mode_rows[mode_name] = deduped
        mode_stats[mode_name] = {
            "before": len(rows),
            "after": len(deduped),
            "removed": len(rows) - len(deduped),
        }

    summary_path = corruption_dir / "corruption_summary.json"
    metadata: dict[str, Any] = {}
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = dict(loaded.get("metadata") or {})
    metadata["dedupe_removed_trace_ids"] = len(removed_trace_ids)
    summary = summarize_corruption_records(mode_rows)
    summary_path.write_text(
        json.dumps({"metadata": metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"modes": mode_stats, "summary_path": str(summary_path)}


def _dedupe_trace_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    deduped: list[dict[str, Any]] = []
    removed_trace_ids: set[str] = set()
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row.get("question_id"),
            row.get("prompt_id"),
            row.get("raw_completion"),
            tuple(row.get("steps", [])),
            row.get("final_answer_line"),
            row.get("extracted_answer"),
            row.get("is_correct"),
            row.get("extraction_failed"),
        )
        if key in seen:
            trace_id = row.get("trace_id")
            if isinstance(trace_id, str):
                removed_trace_ids.add(trace_id)
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed_trace_ids


def _dedupe_rows(
    rows: list[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
