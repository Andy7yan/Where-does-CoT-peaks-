"""One-off conversion from a legacy full_generation run to the canonical difficulty handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.aggregation import aggregate_stage1_outputs


def main() -> None:
    """Convert one legacy run in place into the current canonical handoff layout."""

    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    artifacts = aggregate_stage1_outputs(
        str(run_dir),
        config_path=args.config,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/full_generation",
        help="Legacy full_generation run directory to convert in place.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used for thresholds and handoff construction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
