"""Build the per-question data-phase handoff for a dense PQ run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.per_question_pipeline import aggregate_per_question_outputs


def main() -> None:
    """Run the per-question data-phase export."""

    args = parse_args()
    artifacts = aggregate_per_question_outputs(
        args.run_dir,
        config_path=args.config,
    )
    for key, value in artifacts.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the per-question data-phase export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Per-question run directory containing shards and manifest files.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1_per_question.yaml",
        help="Path to the per-question config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
