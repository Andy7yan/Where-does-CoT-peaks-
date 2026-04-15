"""CLI for running the Stage 1 Pilot workflow."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pilot import run_pilot


def main() -> None:
    """Run the Stage D Pilot workflow."""

    args = parse_args()
    if args.debug:
        os.environ["PEAK_COT_DEBUG"] = "1"

    artifacts = run_pilot(
        config_path=args.config,
        output_dir=args.output_dir,
        source=args.source,
        cache_dir=args.cache_dir,
        local_path=args.local_path,
        prompts_dir=args.prompts_dir,
        mock=args.mock,
        data_path=args.data,
    )

    for key, value in artifacts.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Pilot execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for Pilot outputs.")
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Where to load the configured dataset from in real mode.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional datasets cache directory for Hugging Face loading.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to a local GSM8K-Platinum-style JSON file when using --source local.",
    )
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Directory containing icl_*.yaml prompt templates.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the Pilot workflow with the deterministic local mock generator.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Local GSM8K-Platinum-style JSON file required by --mock.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose generation debug logs.",
    )
    args = parser.parse_args()

    if args.mock and not args.data:
        parser.error("--mock requires --data.")

    return args


if __name__ == "__main__":
    main()
