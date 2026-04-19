"""Repair planning helpers for incomplete per-question generation runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from src.data_phase1.per_question_selection import (
    PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD,
    PER_QUESTION_MANIFEST_FILENAME,
    PER_QUESTION_SELECTION_META_FILENAME,
    infer_per_question_shard_count,
    load_per_question_manifest,
    load_per_question_selection_metadata,
    plan_per_question_shards,
    save_per_question_manifest,
)


REPAIR_DIRNAME = "repair"
REPAIR_MANIFEST_FILENAME = "repair_manifest.jsonl"
REPAIR_SELECTION_META_FILENAME = "repair_selection_meta.json"
REPAIR_REPORT_FILENAME = "repair_report.json"
REPAIR_ISSUES_FILENAME = "repair_issues.jsonl"
REPAIR_SHARD_PLAN_FILENAME = "repair_shards.jsonl"
RUN_SHARD_PLAN_FILENAME = "per_question_shards.jsonl"
_SHARD_ID_PATTERN = re.compile(r"^q(?P<start>\d{4})_(?P<end>\d{4})$")


@dataclass(frozen=True)
class TraceReadResult:
    """Trace counts recovered from one shard trace table."""

    counts_by_question: dict[str, int]
    invalid_line_count: int
    invalid_line_samples: tuple[str, ...]


def build_repair_bundle(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    questions_per_shard: int | None = None,
    target_traces_per_shard: int = PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD,
    include_append_unsafe: bool = False,
    single_shard: bool = True,
    exclude_shard_ids: set[str] | None = None,
    exclude_question_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Scan a per-question run and materialize a reusable repair bundle."""

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Per-question run directory not found: {run_path}")

    output_path = Path(output_dir) if output_dir is not None else run_path / REPAIR_DIRNAME
    output_path.mkdir(parents=True, exist_ok=True)
    excluded_shard_ids = set(exclude_shard_ids or set())
    excluded_question_ids = set(exclude_question_ids or set())

    manifest_rows = load_per_question_manifest(run_path / PER_QUESTION_MANIFEST_FILENAME)
    selection_meta = _load_selection_meta_if_present(run_path)
    shard_plan = _build_expected_shard_plan(
        run_path,
        manifest_rows,
        questions_per_shard=questions_per_shard,
        target_traces_per_shard=target_traces_per_shard,
    )
    global_counts, source_shards_by_question, shard_trace_read = _scan_all_shards(run_path)
    issues, shard_status = _build_issue_reports(
        manifest_rows=manifest_rows,
        shard_plan=shard_plan,
        global_counts=global_counts,
        source_shards_by_question=source_shards_by_question,
        shard_trace_read=shard_trace_read,
        run_path=run_path,
    )

    safe_issue_rows = [row for row in issues if bool(row["append_safe"])]
    unsafe_issue_rows = [row for row in issues if not bool(row["append_safe"])]
    selected_issue_rows = issues if include_append_unsafe else safe_issue_rows
    excluded_issue_rows = [
        row
        for row in selected_issue_rows
        if str(row["expected_shard_id"]) in excluded_shard_ids
        or str(row["question_id"]) in excluded_question_ids
    ]
    selected_issue_rows = [
        row
        for row in selected_issue_rows
        if str(row["expected_shard_id"]) not in excluded_shard_ids
        and str(row["question_id"]) not in excluded_question_ids
    ]
    repair_manifest = [_manifest_row_from_issue(row) for row in selected_issue_rows]

    repair_manifest_path: str | None = None
    repair_meta_path: str | None = None
    repair_shard_plan_path: str | None = None
    repair_shard_count = 0
    if repair_manifest:
        repair_selection_meta = _build_repair_selection_metadata(
            run_path=run_path,
            base_selection_meta=selection_meta,
            repair_issue_rows=selected_issue_rows,
            safe_issue_count=len(safe_issue_rows),
            unsafe_issue_count=len(unsafe_issue_rows),
        )
        repair_manifest_path, repair_meta_path = save_per_question_manifest(
            output_path,
            manifest=repair_manifest,
            selection_metadata=repair_selection_meta,
            manifest_filename=REPAIR_MANIFEST_FILENAME,
            meta_filename=REPAIR_SELECTION_META_FILENAME,
        )
        repair_shard_count = 1 if single_shard else infer_per_question_shard_count(
            repair_manifest,
            questions_per_shard=questions_per_shard,
            target_traces_per_shard=target_traces_per_shard,
        )
        repair_shard_plan = plan_per_question_shards(repair_manifest, shard_count=repair_shard_count)
        repair_shard_plan_path = _write_jsonl(output_path / REPAIR_SHARD_PLAN_FILENAME, repair_shard_plan)
    else:
        repair_shard_plan = []

    issues_path = _write_jsonl(output_path / REPAIR_ISSUES_FILENAME, issues)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path).replace("\\", "/"),
        "output_dir": str(output_path).replace("\\", "/"),
        "manifest_path": str(run_path / PER_QUESTION_MANIFEST_FILENAME).replace("\\", "/"),
        "total_manifest_questions": len(manifest_rows),
        "completed_questions": sum(1 for row in manifest_rows if global_counts.get(str(row["question_id"]), 0) == int(row["target_total_traces"])),
        "issue_count": len(issues),
        "append_safe_issue_count": len(safe_issue_rows),
        "append_unsafe_issue_count": len(unsafe_issue_rows),
        "repair_manifest_count": len(repair_manifest),
        "excluded_issue_count": len(excluded_issue_rows),
        "include_append_unsafe": include_append_unsafe,
        "single_shard": single_shard,
        "exclude_shard_ids": sorted(excluded_shard_ids),
        "exclude_question_ids": sorted(excluded_question_ids),
        "expected_shard_count": len(shard_plan),
        "repair_shard_count": repair_shard_count,
        "repair_manifest_path": _normalize_optional_path(repair_manifest_path),
        "repair_selection_meta_path": _normalize_optional_path(repair_meta_path),
        "repair_shard_plan_path": _normalize_optional_path(repair_shard_plan_path),
        "issues_path": str(issues_path).replace("\\", "/"),
        "shard_status": shard_status,
    }
    report_path = output_path / REPAIR_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path).replace("\\", "/")
    return report


def _scan_all_shards(
    run_path: Path,
) -> tuple[dict[str, int], dict[str, list[str]], dict[str, TraceReadResult]]:
    global_counts: defaultdict[str, int] = defaultdict(int)
    source_shards_by_question: defaultdict[str, list[str]] = defaultdict(list)
    shard_trace_read: dict[str, TraceReadResult] = {}

    shards_root = run_path / "shards"
    if not shards_root.exists():
        return {}, {}, {}

    for shard_dir in sorted(path for path in shards_root.iterdir() if path.is_dir()):
        traces_path = shard_dir / "traces.jsonl"
        if not traces_path.exists():
            shard_trace_read[shard_dir.name] = TraceReadResult({}, 0, tuple())
            continue
        read_result = _read_trace_counts(traces_path)
        shard_trace_read[shard_dir.name] = read_result
        for question_id, count in read_result.counts_by_question.items():
            global_counts[question_id] += count
            source_shards_by_question[question_id].append(shard_dir.name)

    return dict(global_counts), dict(source_shards_by_question), shard_trace_read


def _read_trace_counts(traces_path: Path) -> TraceReadResult:
    counts: defaultdict[str, int] = defaultdict(int)
    invalid_line_count = 0
    invalid_line_samples: list[str] = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_line_count += 1
                if len(invalid_line_samples) < 3:
                    invalid_line_samples.append(stripped[:200])
                continue
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                continue
            counts[question_id] += 1
    return TraceReadResult(
        counts_by_question=dict(counts),
        invalid_line_count=invalid_line_count,
        invalid_line_samples=tuple(invalid_line_samples),
    )


def _build_issue_reports(
    *,
    manifest_rows: list[dict[str, Any]],
    shard_plan: list[dict[str, int]],
    global_counts: dict[str, int],
    source_shards_by_question: dict[str, list[str]],
    shard_trace_read: dict[str, TraceReadResult],
    run_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    shard_status: list[dict[str, Any]] = []

    for shard in shard_plan:
        shard_id = _format_shard_id(shard["start_idx"], shard["end_idx"])
        expected_rows = manifest_rows[shard["start_idx"] : shard["end_idx"]]
        shard_dir = run_path / "shards" / shard_id
        read_result = shard_trace_read.get(shard_id, TraceReadResult({}, 0, tuple()))
        log_excerpt = _extract_log_error_excerpt(run_path / "logs" / f"generation-{shard_id}.log")
        shard_missing = 0
        shard_mismatch = 0
        for manifest_index, row in enumerate(expected_rows, start=shard["start_idx"]):
            question_id = str(row["question_id"])
            expected_count = int(row["target_total_traces"])
            observed_count = int(global_counts.get(question_id, 0))
            if observed_count == expected_count:
                continue
            issue_type = "missing_question" if observed_count == 0 else "trace_count_mismatch"
            append_safe = observed_count == 0
            if observed_count == 0:
                shard_missing += 1
            else:
                shard_mismatch += 1
            issues.append(
                {
                    "question_id": question_id,
                    "manifest_index": manifest_index,
                    "expected_shard_id": shard_id,
                    "issue_type": issue_type,
                    "append_safe": append_safe,
                    "source_difficulty_bucket": str(row["source_difficulty_bucket"]),
                    "target_total_traces": expected_count,
                    "observed_trace_count": observed_count,
                    "source_shards": list(source_shards_by_question.get(question_id, [])),
                    "question_text": str(row["question_text"]),
                    "gold_answer": row["gold_answer"],
                }
            )

        shard_status.append(
            {
                "shard_id": shard_id,
                "start_idx": shard["start_idx"],
                "end_idx": shard["end_idx"],
                "question_count": shard["question_count"],
                "target_total_traces": shard["target_total_traces"],
                "shard_dir_exists": shard_dir.exists(),
                "questions_missing": shard_missing,
                "questions_mismatched": shard_mismatch,
                "invalid_trace_lines": read_result.invalid_line_count,
                "invalid_trace_line_samples": list(read_result.invalid_line_samples),
                "log_error_excerpt": log_excerpt,
            }
        )

    return issues, shard_status


def _build_expected_shard_plan(
    run_path: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    questions_per_shard: int | None,
    target_traces_per_shard: int,
) -> list[dict[str, int]]:
    explicit_plan = _load_expected_shard_plan_from_run(run_path, manifest_rows)
    if explicit_plan:
        return explicit_plan
    shard_count = infer_per_question_shard_count(
        manifest_rows,
        questions_per_shard=questions_per_shard,
        target_traces_per_shard=target_traces_per_shard,
    )
    return plan_per_question_shards(manifest_rows, shard_count=shard_count)


def _load_expected_shard_plan_from_run(
    run_path: Path,
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, int]]:
    shard_plan_path = run_path / RUN_SHARD_PLAN_FILENAME
    if shard_plan_path.exists():
        rows = []
        for line in shard_plan_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            start_idx = int(row["start_idx"])
            end_idx = int(row["end_idx"])
            rows.append(
                {
                    "shard_index": len(rows),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "question_count": end_idx - start_idx,
                    "target_total_traces": _sum_target_traces(manifest_rows, start_idx, end_idx),
                }
            )
        if rows:
            return rows

    shard_ranges: set[tuple[int, int]] = set()
    logs_dir = run_path / "logs"
    found_log_ranges = False
    if logs_dir.exists():
        for log_path in logs_dir.glob("generation-q*.log"):
            parsed = _parse_shard_id(log_path.stem.removeprefix("generation-"))
            if parsed is not None:
                found_log_ranges = True
                shard_ranges.add(parsed)

    if found_log_ranges:
        shards_dir = run_path / "shards"
        if shards_dir.exists():
            for shard_dir in shards_dir.iterdir():
                if not shard_dir.is_dir():
                    continue
                parsed = _parse_shard_id(shard_dir.name)
                if parsed is None:
                    continue
                shard_ranges.add(parsed)

    rows = []
    for start_idx, end_idx in sorted(shard_ranges):
        rows.append(
            {
                "shard_index": len(rows),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "question_count": end_idx - start_idx,
                "target_total_traces": _sum_target_traces(manifest_rows, start_idx, end_idx),
            }
        )
    return rows


def _build_repair_selection_metadata(
    *,
    run_path: Path,
    base_selection_meta: dict[str, Any],
    repair_issue_rows: list[dict[str, Any]],
    safe_issue_count: int,
    unsafe_issue_count: int,
) -> dict[str, Any]:
    metadata = dict(base_selection_meta)
    metadata["selected_question_count"] = len(repair_issue_rows)
    metadata["repair_source_run_dir"] = str(run_path).replace("\\", "/")
    metadata["repair_generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata["repair_issue_summary"] = {
        "append_safe_issue_count": safe_issue_count,
        "append_unsafe_issue_count": unsafe_issue_count,
    }
    metadata["repair_issue_types"] = sorted({str(row["issue_type"]) for row in repair_issue_rows})
    return metadata


def _load_selection_meta_if_present(run_path: Path) -> dict[str, Any]:
    selection_meta_path = run_path / PER_QUESTION_SELECTION_META_FILENAME
    if selection_meta_path.exists():
        return load_per_question_selection_metadata(selection_meta_path)
    return {
        "schema_version": "per_question_selection_v1",
        "pipeline_variant": "per_question",
        "selected_question_count": 0,
    }


def _manifest_row_from_issue(issue_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": str(issue_row["question_id"]),
        "question_text": str(issue_row["question_text"]),
        "gold_answer": issue_row["gold_answer"],
        "source_difficulty_bucket": str(issue_row["source_difficulty_bucket"]),
        "target_total_traces": int(issue_row["target_total_traces"]),
        "target_samples_per_prompt": _target_samples_from_traces(int(issue_row["target_total_traces"])),
    }


def _target_samples_from_traces(target_total_traces: int) -> int:
    if target_total_traces == 120:
        return 30
    if target_total_traces == 300:
        return 75
    raise ValueError(f"Unsupported target_total_traces for repair manifest: {target_total_traces}")


def _extract_log_error_excerpt(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    lines = [line.rstrip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "Traceback" in stripped or stripped.startswith(
            (
                "json.decoder.",
                "ValueError",
                "RuntimeError",
                "FileNotFoundError",
                "KeyError",
                "TypeError",
            )
        ):
            return stripped
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _format_shard_id(start_idx: int, end_idx: int) -> str:
    return f"q{start_idx:04d}_{end_idx:04d}"


def _parse_shard_id(value: str) -> tuple[int, int] | None:
    match = _SHARD_ID_PATTERN.fullmatch(value)
    if match is None:
        return None
    return int(match.group("start")), int(match.group("end"))


def _sum_target_traces(
    manifest_rows: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> int:
    return sum(int(row["target_total_traces"]) for row in manifest_rows[start_idx:end_idx])


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\", "/")


__all__ = [
    "REPAIR_DIRNAME",
    "REPAIR_ISSUES_FILENAME",
    "REPAIR_MANIFEST_FILENAME",
    "REPAIR_REPORT_FILENAME",
    "REPAIR_SELECTION_META_FILENAME",
    "REPAIR_SHARD_PLAN_FILENAME",
    "build_repair_bundle",
]
