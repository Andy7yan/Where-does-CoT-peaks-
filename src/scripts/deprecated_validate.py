"""Validate the canonical Stage 1 data-phase artifacts before analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.curation import validate_canonical_data_phase


def main() -> None:
    """Run validation checks for the canonical analysis input directory."""

    args = parse_args()
    run_path = Path(args.canonical_run_dir)
    validation = validate_canonical_data_phase(run_path, config_path=args.config)
    if args.json:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    print(f"canonical_run_dir: {run_path}")
    print(f"trace_count: {validation['trace_count']}")
    print(f"question_count: {validation['question_count']}")
    print(f"trace_id_unique: {validation['trace_id_unique']}")
    print(
        "question_metadata_matches_traces: "
        f"{validation['question_metadata_matches_traces']}"
    )
    print(f"accuracy_csv_matches_traces: {validation['accuracy_csv_matches_traces']}")
    print(
        "corruption_summary_matches_records: "
        f"{validation['corruption_summary_matches_records']}"
    )
    print(
        "difficulty_exports_match_traces: "
        f"{validation['difficulty_exports_match_traces']}"
    )
    for mode_name, summary in validation["corruption_validation"].items():
        print(
            f"{mode_name}: records={summary['records']} "
            f"failures={summary['failures']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-run-dir",
        required=True,
        help="Canonical deduplicated run directory used as the analysis entrypoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Path to the experiment config YAML used for validation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation payload as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
