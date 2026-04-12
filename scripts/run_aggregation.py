"""CLI for Stage 1 aggregation tasks."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reports import aggregate_stage1_outputs


def main() -> None:
    """Run a Stage 1 aggregation task."""

    args = parse_args()
    if args.stage == "e":
        artifacts = aggregate_stage1_outputs(
            args.run_dir,
            config_path=args.config,
        )
        for key, value in artifacts.items():
            print(f"{key}: {value}")
        return

    raise NotImplementedError("Stage G aggregation is not implemented yet.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for aggregation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("e", "g"),
        required=True,
        help="Aggregation stage to run.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing Stage inputs and receiving outputs.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Path to the experiment config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
