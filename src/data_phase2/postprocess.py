"""Postprocess existing runs into the canonical handoff and downstream analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis_phase1.analysis import run_analysis
from src.analysis_phase1.backend import load_analysis_backend
from src.common.settings import ExperimentConfig
from src.data_phase2.aggregation import aggregate_stage1_outputs
from src.data_phase2.curation import validate_canonical_data_phase


def run_postprocess_pipeline(
    *,
    run_dir: str | Path,
    config_path: str = "configs/stage1.yaml",
    include_analysis: bool = True,
) -> dict[str, Any]:
    """Convert one existing run into the canonical handoff and optionally run analysis."""

    run_path = Path(run_dir)
    aggregation = aggregate_stage1_outputs(str(run_path), config_path=config_path)
    validation = validate_canonical_data_phase(run_path, config_path=config_path)

    result: dict[str, Any] = {
        "run_dir": str(run_path),
        "aggregation": aggregation,
        "validation": validation,
    }
    if not include_analysis:
        return result

    config = ExperimentConfig.from_yaml(config_path)
    backend = load_analysis_backend(config)
    analysis = run_analysis(
        run_dir=str(run_path),
        prompt_logits_fn=backend["prompt_logits_fn"],
        prompt_logits_batch_fn=backend.get("prompt_logits_batch_fn"),
        prompt_measurement_fn=backend.get("prompt_measurement_fn"),
        prompt_measurement_batch_fn=backend.get("prompt_measurement_batch_fn"),
        tokenizer=backend["tokenizer"],
        trace_trajectory_fn=backend["trace_trajectory_fn"],
        ld_epsilon=config.nldd.ld_epsilon,
        tas_plateau_threshold=config.tas.plateau_threshold,
        perplexity_filter_enabled=config.nldd.perplexity_filter_enabled,
        perplexity_ratio_threshold=config.nldd.perplexity_ratio_threshold,
    )
    result["analysis"] = analysis
    result["runtime_selection"] = {
        "requested_device": backend["runtime_selection"].requested_device,
        "resolved_device": backend["runtime_selection"].resolved_device,
        "reason": backend["runtime_selection"].reason,
        "gpu_name": backend["runtime_selection"].gpu_name,
        "gpu_compute_capability": backend["runtime_selection"].gpu_compute_capability,
    }
    return result


__all__ = ["run_postprocess_pipeline"]
