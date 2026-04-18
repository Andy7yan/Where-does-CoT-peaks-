"""Run the analysis workflow on the canonical difficulty/sample layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase1.analysis import run_analysis
from src.analysis_phase1.backend import load_analysis_backend
from src.common.settings import ExperimentConfig


def main() -> None:
    """Run the analysis pipeline and persist analysis outputs."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    backend = load_analysis_backend(config)
    artifacts = run_analysis(
        run_dir=args.run_dir,
        prompt_logits_fn=backend["prompt_logits_fn"],
        tokenizer=backend["tokenizer"],
        trace_trajectory_fn=backend["trace_trajectory_fn"],
        ld_epsilon=config.nldd.ld_epsilon,
        tas_plateau_threshold=config.tas.plateau_threshold,
    )

    print(f"runtime_device_requested: {backend['runtime_selection'].requested_device}")
    print(f"runtime_device_resolved: {backend['runtime_selection'].resolved_device}")
    print(f"analysis_phase1_dir: {artifacts['analysis_dir']}")
    print(f"sample_count: {artifacts['sample_count']}")
    print(f"s_value: {artifacts['s_value']}")
    print(f"accuracy_by_length_path: {artifacts['accuracy_by_length_path']}")
    print(f"nldd_per_trace_path: {artifacts['nldd_per_trace_path']}")
    print(f"tas_per_trace_path: {artifacts['tas_per_trace_path']}")
    print(f"tas_curve_per_trace_path: {artifacts['tas_curve_per_trace_path']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Canonical run directory rooted at a difficulty/bin/sample layout.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used to resolve measurement settings.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
