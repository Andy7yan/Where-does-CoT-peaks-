"""Export all traces for qualified per-question questions without touching source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PQ_RUN_DIR = Path("results/per-question-0419_143551")
DEFAULT_OUTPUT_DIR = Path("results/final_outputs/qualified_traces")
PQ_ANALYSIS_DIRNAME = "pq_analysis"


def main() -> None:
    args = parse_args()
    pq_run_dir = Path(args.pq_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qualified_ids = load_qualified_question_ids(pq_run_dir / PQ_ANALYSIS_DIRNAME / "qualified_questions_full_metrics.csv")
    traces_path = pq_run_dir / "traces.jsonl"
    metadata_path = pq_run_dir / "question_metadata.jsonl"

    qualified_traces = [
        row
        for row in _load_jsonl(traces_path)
        if str(row.get("question_id")) in qualified_ids
    ]
    qualified_metadata = [
        row
        for row in _load_jsonl(metadata_path)
        if str(row.get("question_id")) in qualified_ids
    ]

    qualified_traces_path = output_dir / "qualified_traces.jsonl"
    qualified_metadata_path = output_dir / "qualified_question_metadata.jsonl"
    qualified_ids_path = output_dir / "qualified_question_ids.txt"
    summary_path = output_dir / "qualified_trace_export_summary.json"

    _write_jsonl(qualified_traces_path, qualified_traces)
    _write_jsonl(qualified_metadata_path, qualified_metadata)
    qualified_ids_path.write_text("".join(f"{question_id}\n" for question_id in sorted(qualified_ids)), encoding="utf-8")

    trace_counts_by_question: dict[str, int] = {}
    for row in qualified_traces:
        question_id = str(row["question_id"])
        trace_counts_by_question[question_id] = trace_counts_by_question.get(question_id, 0) + 1

    summary = {
        "pq_run_dir": str(pq_run_dir),
        "source_traces_path": str(traces_path),
        "source_question_metadata_path": str(metadata_path),
        "qualified_question_count": len(qualified_ids),
        "qualified_trace_count": len(qualified_traces),
        "qualified_metadata_count": len(qualified_metadata),
        "output_files": {
            "qualified_traces_path": str(qualified_traces_path),
            "qualified_question_metadata_path": str(qualified_metadata_path),
            "qualified_question_ids_path": str(qualified_ids_path),
        },
        "per_question_trace_counts_top20": [
            {"question_id": question_id, "trace_count": trace_counts_by_question[question_id]}
            for question_id in sorted(
                trace_counts_by_question,
                key=lambda item: (-trace_counts_by_question[item], item),
            )[:20]
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"qualified_question_count: {len(qualified_ids)}")
    print(f"qualified_trace_count: {len(qualified_traces)}")
    print(f"qualified_traces_path: {qualified_traces_path}")
    print(f"qualified_question_metadata_path: {qualified_metadata_path}")
    print(f"qualified_question_ids_path: {qualified_ids_path}")
    print(f"summary_path: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pq-run-dir",
        default=str(DEFAULT_PQ_RUN_DIR),
        help="Per-question run directory containing traces.jsonl and pq_analysis/qualified_questions_full_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the frozen qualified-only exports.",
    )
    return parser.parse_args()


def load_qualified_question_ids(path: Path) -> set[str]:
    rows = _load_csv(path)
    return {
        str(row["question_id"])
        for row in rows
        if _parse_bool(row.get("degenerate")) is False
        and _parse_bool(row.get("l_curve_insufficient")) is False
        and _parse_bool(row.get("k_star_insufficient")) is False
    }


def _load_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


if __name__ == "__main__":
    main()
