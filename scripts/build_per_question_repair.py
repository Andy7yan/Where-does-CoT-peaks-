"""Scan a per-question run and build a reusable repair manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.per_question_repair import build_repair_bundle


def main() -> None:
    args = parse_args()
    report = build_repair_bundle(
        args.run_dir,
        output_dir=args.output_dir,
        questions_per_shard=args.questions_per_shard,
        target_traces_per_shard=args.target_traces_per_shard,
        include_append_unsafe=args.include_append_unsafe,
        single_shard=not args.multi_shard,
        exclude_shard_ids=set(args.exclude_shard_id),
        exclude_question_ids=set(args.exclude_question_id),
    )
    for key in (
        "run_dir",
        "output_dir",
        "total_manifest_questions",
        "completed_questions",
        "issue_count",
        "append_safe_issue_count",
        "append_unsafe_issue_count",
        "excluded_issue_count",
        "repair_manifest_count",
        "repair_manifest_path",
        "repair_shard_plan_path",
        "report_path",
    ):
        print(f"{key}: {report.get(key)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Per-question run directory to scan.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory receiving the generated repair bundle. Defaults to <run-dir>/repair.",
    )
    parser.add_argument(
        "--questions-per-shard",
        type=int,
        default=None,
        help="Optional shard budget used when inferring the expected original shard plan.",
    )
    parser.add_argument(
        "--target-traces-per-shard",
        type=int,
        default=7200,
        help="Target traces per shard used when inferring the expected original shard plan.",
    )
    parser.add_argument(
        "--include-append-unsafe",
        action="store_true",
        help="Also include trace-count-mismatch questions in the repair manifest.",
    )
    parser.add_argument(
        "--multi-shard",
        action="store_true",
        help="Build a balanced repair shard plan instead of collapsing all repair questions into one shard.",
    )
    parser.add_argument(
        "--exclude-shard-id",
        action="append",
        default=[],
        help="Original shard id to exclude from the generated repair manifest. Repeat as needed.",
    )
    parser.add_argument(
        "--exclude-question-id",
        action="append",
        default=[],
        help="Question id to exclude from the generated repair manifest. Repeat as needed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
