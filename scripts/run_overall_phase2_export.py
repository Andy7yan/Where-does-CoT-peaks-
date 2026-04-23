"""Materialize the compact overall Phase 2 analysis views from overall analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase2.plotting import ensure_overall_analysis_views


def main() -> None:
    args = parse_args()
    analysis_dir = ensure_overall_analysis_views(args.run_dir)
    print(f"analysis_dir: {analysis_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Overall run directory containing analysis_phase1/ or analysis/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
