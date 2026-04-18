"""Convert an existing run into the canonical handoff and downstream analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.postprocess import run_postprocess_pipeline


def main() -> None:
    """Run the postprocess pipeline on an existing run directory."""

    args = parse_args()
    result = run_postprocess_pipeline(
        run_dir=args.run_dir,
        config_path=args.config,
        include_analysis=not args.skip_analysis,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory with traces and old or new corruption artifacts.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used for canonicalization and analysis.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only rebuild the canonical handoff and validation artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
