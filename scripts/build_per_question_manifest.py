"""Build the frozen medium/hard question manifest for the per-question path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.per_question_selection import (
    build_per_question_manifest,
    build_per_question_selection_metadata,
    save_per_question_manifest,
)


def main() -> None:
    """Build and persist the per-question manifest."""

    args = parse_args()
    manifest = build_per_question_manifest(
        config_path=args.config,
        source_run=args.source_run,
        source=args.source,
        local_path=args.local_path,
        cache_dir=args.cache_dir,
    )
    selection_metadata = build_per_question_selection_metadata(
        config_path=args.config,
        source_run=args.source_run,
        manifest=manifest,
    )
    manifest_path, selection_meta_path = save_per_question_manifest(
        args.output_dir,
        manifest=manifest,
        selection_metadata=selection_metadata,
    )

    print(f"manifest_path: {manifest_path}")
    print(f"selection_meta_path: {selection_meta_path}")
    print(f"selected_question_count: {selection_metadata['selected_question_count']}")
    print(f"difficulty_counts: {selection_metadata['difficulty_counts']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for per-question manifest building."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the per-question config YAML.")
    parser.add_argument(
        "--source-run",
        required=True,
        help="Source run directory or run name providing question_metadata.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory receiving the manifest and selection metadata.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Where to load the configured dataset from.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to a local GSM8K-compatible JSON or JSONL file when using --source local.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional datasets cache directory for Hugging Face loading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
