"""Profile question difficulty before choosing easy/medium/hard boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.difficulty_profile import export_difficulty_profile


def main() -> None:
    args = parse_args()
    result = export_difficulty_profile(
        run_dir=args.run_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        bin_size=args.bin_size,
        write_plot=not args.no_plot,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory with traces or shard traces.")
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used only for reporting current threshold defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/difficulty_profile.",
    )
    parser.add_argument("--bin-size", type=float, default=0.05, help="Histogram bin width.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot rendering.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
