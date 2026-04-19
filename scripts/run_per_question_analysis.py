"""Run per-question analysis and write pq_analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase1.backend import load_analysis_backend
from src.analysis_phase1.per_question_analysis import run_per_question_analysis
from src.common.settings import ExperimentConfig


def main() -> None:
    """Run the per-question analysis pipeline and persist pq_analysis outputs."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    backend = load_analysis_backend(config)
    min_kstar_bins = (
        2
        if config.analysis.per_question_min_kstar_bins is None
        else int(config.analysis.per_question_min_kstar_bins)
    )
    artifacts = run_per_question_analysis(
        run_dir=args.run_dir,
        prompt_logits_fn=backend["prompt_logits_fn"],
        tokenizer=backend["tokenizer"],
        trace_trajectory_fn=backend["trace_trajectory_fn"],
        ld_epsilon=config.nldd.ld_epsilon,
        tas_plateau_threshold=config.tas.plateau_threshold,
        min_kstar_bins=min_kstar_bins,
    )

    print(f"runtime_device_requested: {backend['runtime_selection'].requested_device}")
    print(f"runtime_device_resolved: {backend['runtime_selection'].resolved_device}")
    print(f"pq_analysis_dir: {artifacts['analysis_dir']}")
    print(f"sample_count: {artifacts['sample_count']}")
    print(f"s_value: {artifacts['s_value']}")
    print(f"t1b_step_surface_path: {artifacts['t1b_step_surface_path']}")
    print(f"t1c_kstar_ratio_path: {artifacts['t1c_kstar_ratio_path']}")
    print(f"t2b_lstar_difficulty_path: {artifacts['t2b_lstar_difficulty_path']}")
    print(f"bin_status_path: {artifacts['bin_status_path']}")
    print(f"failure_stats_path: {artifacts['failure_stats_path']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the per-question analysis pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Per-question run directory containing the per_question handoff tree.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1_per_question.yaml",
        help="Path to the per-question config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
