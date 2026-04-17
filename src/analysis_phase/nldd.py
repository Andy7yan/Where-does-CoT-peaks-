"""Backward-compatible NLDD facade re-exporting specialized workflow modules."""

from src.analysis_phase.nldd_corruption import (
    CorruptionSelectionConfig,
    build_corruption_record,
    build_corruption_records,
    load_trace_sources,
    summarize_corruption_records,
    write_corruption_artifacts,
)
from src.analysis_phase.nldd_measurement import (
    build_correct_token_ids,
    build_prompt_logit_fn,
    calibrate_s,
    compute_logit_margin,
    compute_v4_measurement_artifacts,
    extract_trace_horizon,
    measure_nldd,
    measure_selected_traces,
    measure_trace_profile,
    validate_nldd_full_records,
    write_nldd_full_records,
    write_s_calibration,
)
from src.analysis_phase.nldd_selection import (
    TraceSelectionConfig,
    build_v4_trace_selection,
    load_coarse_analysis,
    load_or_build_trace_selection,
    load_question_metadata,
    load_trace_selection,
    write_trace_selection,
)
from src.analysis_phase.nldd_shared import (
    TRACE_SELECTION_FIELDNAMES,
    TRACE_SELECTION_REQUIRED_COLUMNS,
)

__all__ = [
    "CorruptionSelectionConfig",
    "TRACE_SELECTION_FIELDNAMES",
    "TRACE_SELECTION_REQUIRED_COLUMNS",
    "TraceSelectionConfig",
    "build_correct_token_ids",
    "build_corruption_record",
    "build_corruption_records",
    "build_prompt_logit_fn",
    "build_v4_trace_selection",
    "calibrate_s",
    "compute_logit_margin",
    "compute_v4_measurement_artifacts",
    "extract_trace_horizon",
    "load_coarse_analysis",
    "load_or_build_trace_selection",
    "load_question_metadata",
    "load_trace_selection",
    "load_trace_sources",
    "measure_nldd",
    "measure_selected_traces",
    "measure_trace_profile",
    "summarize_corruption_records",
    "validate_nldd_full_records",
    "write_corruption_artifacts",
    "write_nldd_full_records",
    "write_s_calibration",
    "write_trace_selection",
]
