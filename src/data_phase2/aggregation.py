"""Backward-compatible facade for data-phase aggregation helpers and pipeline entrypoints."""

from src.data_phase2.aggregation_core import (
    AccuracyBucket,
    build_accuracy_buckets,
    build_question_metadata,
    choose_merge_neighbor,
    merge_sparse_accuracy_buckets,
    select_l_star,
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
    "AccuracyBucket",
    "aggregate_stage1_outputs",
    "build_accuracy_buckets",
    "build_question_metadata",
    "choose_merge_neighbor",
    "discover_stage1_shard_paths",
    "ensure_root_run_metadata",
    "load_stage1_traces",
    "merge_sparse_accuracy_buckets",
    "merge_stage1_shards",
    "plot_stage1_figures",
    "select_l_star",
]
