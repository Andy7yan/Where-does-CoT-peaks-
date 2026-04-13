"""Curate Stage 1 data-phase artifacts into a canonical analysis entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase import curate_data_phase


def main() -> None:
    args = parse_args()
    result = curate_data_phase(
        args.canonical_run_dir,
        legacy_run_dir=args.legacy_run_dir,
        config_path=args.config,
    )
    print(f"canonical_run_dir: {result['canonical_run_dir']}")
    print(f"legacy_run_dir: {result['legacy_run_dir']}")
    print(f"manifest_path: {result['manifest_path']}")
    print(f"readme_path: {result['readme_path']}")
    print(f"legacy_readme_path: {result['legacy_readme_path']}")
    print(f"moved_items: {len(result['moved_items'])}")
    print(f"trace_count: {result['validation']['trace_count']}")
    print(f"question_count: {result['validation']['question_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-run-dir",
        required=True,
        help="Canonical deduplicated run directory used as the analysis entrypoint.",
    )
    parser.add_argument(
        "--legacy-run-dir",
        required=True,
        help="Legacy historical run directory retained for provenance only.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Path to the experiment config YAML used for validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
