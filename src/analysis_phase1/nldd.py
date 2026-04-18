"""Compatibility facade for NLDD helpers split across selection/corruption/measurement modules."""

from src.analysis_phase1.nldd_corruption import (
    CorruptionSelectionConfig,
    build_corruption_record,
    build_corruption_records,
    load_trace_sources,
    summarize_corruption_records,
    write_corruption_artifacts,
)
from src.analysis_phase1.nldd_measurement import (
    build_correct_token_ids,
    build_prompt_logit_fn,
    compute_logit_margin,
    compute_v4_measurement_artifacts,
    extract_trace_horizon,
    measure_nldd,
    measure_trace_profile,
    validate_nldd_full_records,
)
from src.analysis_phase1.nldd_selection import (
    TraceSelectionConfig,
    load_coarse_analysis,
    load_or_build_trace_selection,
    load_question_metadata,
)

__all__ = [
    "CorruptionSelectionConfig",
    "TraceSelectionConfig",
    "build_correct_token_ids",
    "build_corruption_record",
    "build_corruption_records",
    "build_prompt_logit_fn",
    "compute_logit_margin",
    "compute_v4_measurement_artifacts",
    "extract_trace_horizon",
    "load_coarse_analysis",
    "load_or_build_trace_selection",
    "load_question_metadata",
    "load_trace_sources",
    "measure_nldd",
    "measure_trace_profile",
    "summarize_corruption_records",
    "validate_nldd_full_records",
    "write_corruption_artifacts",
]
