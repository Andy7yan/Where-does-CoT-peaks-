"""Facade for the active data-phase aggregation helpers and entrypoints."""

from src.data_phase2.aggregation_core import (
    build_accuracy_rows,
    select_l_star_from_accuracy_rows,
    standard_error,
)
from src.data_phase2.pipeline import (
    aggregate_stage1_outputs,
    discover_stage1_shard_paths,
    ensure_root_run_metadata,
    load_stage1_traces,
    merge_stage1_shards,
    plot_stage1_figures,
)

__all__ = [
    "aggregate_stage1_outputs",
    "build_accuracy_rows",
    "discover_stage1_shard_paths",
    "ensure_root_run_metadata",
    "load_stage1_traces",
    "merge_stage1_shards",
    "plot_stage1_figures",
    "select_l_star_from_accuracy_rows",
    "standard_error",
]
