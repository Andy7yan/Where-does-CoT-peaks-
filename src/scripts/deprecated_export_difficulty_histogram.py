"""Export a CSV difficulty histogram from question_metadata.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.difficulty_histogram import export_difficulty_histogram


def main() -> None:
    args = parse_args()
    output_path = export_difficulty_histogram(
        question_metadata_path=args.question_metadata,
        output_path=args.output,
        bin_size=args.bin_size,
    )
    print(f"wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-metadata",
        required=True,
        help="Path to question_metadata.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Histogram bin width over [0, 1]. Defaults to 0.05.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
