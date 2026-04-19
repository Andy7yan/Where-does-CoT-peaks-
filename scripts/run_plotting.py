"""Render the static Stage 1 v8 plots for overall and PQ pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase2.plotting import run_stage1_plotting


def main() -> None:
    """Run the static plotting pipeline."""

    args = parse_args()
    representative_questions = (
        [item.strip() for item in args.representative_questions.split(",") if item.strip()]
        if args.representative_questions
        else None
    )
    artifacts = run_stage1_plotting(
        overall_run_dir=args.overall_run_dir,
        pq_run_dir=args.pq_run_dir,
        output_dir=args.output_dir,
        representative_questions=representative_questions,
        max_heatmap_questions=args.max_heatmap_questions,
        normalized_bins=args.normalized_bins,
    )
    for key, value in artifacts.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for static plot rendering."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overall-run-dir",
        required=True,
        help="Run directory for the overall v6 pipeline.",
    )
    parser.add_argument(
        "--pq-run-dir",
        required=True,
        help="Run directory for the dense per-question pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory receiving rendered plot files.",
    )
    parser.add_argument(
        "--representative-questions",
        default=None,
        help="Optional comma-separated list of PQ question_ids for T1-B and T1-B-norm.",
    )
    parser.add_argument(
        "--max-heatmap-questions",
        type=int,
        default=5,
        help="Maximum number of representative PQ questions when auto-selecting them.",
    )
    parser.add_argument(
        "--normalized-bins",
        type=int,
        default=6,
        help="Number of x-axis bins for the normalized T1-B heatmaps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
