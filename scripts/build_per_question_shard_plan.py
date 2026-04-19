"""Build a balanced shard plan for per-question generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.per_question_selection import (
    PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD,
    infer_per_question_shard_count,
    load_per_question_manifest,
    plan_per_question_shards,
    slice_per_question_manifest,
)


def main() -> None:
    """Build and optionally persist a balanced shard plan."""

    args = parse_args()
    manifest_rows = load_per_question_manifest(args.question_manifest)
    selected_rows = slice_per_question_manifest(
        manifest_rows,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    shard_count = infer_per_question_shard_count(
        selected_rows,
        questions_per_shard=args.questions_per_shard,
        target_traces_per_shard=args.target_traces_per_shard,
    )
    shard_plan = plan_per_question_shards(selected_rows, shard_count=shard_count)
    if args.start_idx:
        for row in shard_plan:
            row["start_idx"] += args.start_idx
            row["end_idx"] += args.start_idx

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in shard_plan:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.format == "tsv":
        for row in shard_plan:
            print(
                "\t".join(
                    [
                        str(row["shard_index"]),
                        str(row["start_idx"]),
                        str(row["end_idx"]),
                        str(row["question_count"]),
                        str(row["target_total_traces"]),
                    ]
                )
            )
        return

    print(f"question_manifest_path: {args.question_manifest}")
    print(f"total_questions: {len(selected_rows)}")
    print(f"total_target_traces: {sum(int(row['target_total_traces']) for row in selected_rows)}")
    print(f"shard_count: {len(shard_plan)}")
    print(
        "question_count_range: "
        f"{min(row['question_count'] for row in shard_plan)}-"
        f"{max(row['question_count'] for row in shard_plan)}"
    )
    print(
        "target_trace_range: "
        f"{min(row['target_total_traces'] for row in shard_plan)}-"
        f"{max(row['target_total_traces'] for row in shard_plan)}"
    )
    if args.output_path:
        print(f"shard_plan_path: {args.output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for balanced per-question shard planning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-manifest",
        required=True,
        help="Path to the per-question manifest JSONL.",
    )
    parser.add_argument(
        "--questions-per-shard",
        type=int,
        default=None,
        help="Optional legacy shard budget used only to infer the shard count.",
    )
    parser.add_argument(
        "--target-traces-per-shard",
        type=int,
        default=PER_QUESTION_DEFAULT_TARGET_TRACES_PER_SHARD,
        help="Target total traces per shard when questions-per-shard is not set.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSONL path receiving the computed shard plan.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Inclusive manifest start index for the planned submission window.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Exclusive manifest end index for the planned submission window.",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "tsv"),
        default="summary",
        help="Whether to print only the summary or also emit tab-separated shard rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
