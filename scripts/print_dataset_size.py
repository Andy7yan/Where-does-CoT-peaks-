"""Print the size of the ranked full dataset used for formal generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import ExperimentConfig
from src.data_phase1.tasks import load_question_records_for_config


def main() -> None:
    """Resolve the full ranked question count and print it as an integer."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    ranked_questions = load_question_records_for_config(
        config=config,
        source=args.source,
        local_path=args.local_path,
        cache_dir=args.cache_dir,
    )
    print(len(ranked_questions))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the dataset-size helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Where to load the configured dataset from.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to a local task-compatible JSON or JSONL file when using --source local.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional datasets cache directory for Hugging Face loading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
