"""Run the v4 Stage D2 NLDD measurement workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.generation import _load_tokenizer_with_fallback, _resolve_torch_dtype, ensure_model_available
from src.analysis_phase.nldd import (
    TraceSelectionConfig,
    build_prompt_logit_fn,
    compute_v4_measurement_artifacts,
    load_coarse_analysis,
    load_or_build_trace_selection,
    load_question_metadata,
)
from src.data_phase1.pilot import build_token_counter
from src.data_phase2.aggregation import load_stage1_traces
from src.common.runtime_env import select_runtime_device
from src.common.settings import ExperimentConfig, require_config_value


def main() -> None:
    """Run Stage D2 and persist v4 NLDD measurement artifacts."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_path = Path(args.run_dir)

    traces = load_stage1_traces(run_path)
    question_metadata = load_question_metadata(run_path / "question_metadata.jsonl")
    coarse_analysis = load_coarse_analysis(run_path / "coarse_analysis.json")
    selection_config = TraceSelectionConfig(
        target_traces_per_cell=require_config_value(
            "analysis.target_traces_per_cell",
            config.analysis.target_traces_per_cell,
        ),
        target_traces_near_lstar=require_config_value(
            "analysis.target_traces_near_lstar",
            config.analysis.target_traces_near_lstar,
        ),
        per_question_trace_cap=require_config_value(
            "analysis.per_question_trace_cap",
            config.analysis.per_question_trace_cap,
        ),
        min_nldd_length=require_config_value(
            "analysis.min_nldd_length",
            config.analysis.min_nldd_length,
        ),
        seed=args.seed if args.seed is not None else config.experiment.seed,
    )
    selection_rows, selection_source = load_or_build_trace_selection(
        run_dir=str(run_path),
        traces=traces,
        question_metadata=question_metadata,
        coarse_analysis=coarse_analysis,
        selection_config=selection_config,
    )

    backend = load_measurement_backend(config)
    token_counter = build_token_counter(tokenizer=backend["tokenizer"], approximate=False)
    artifacts = compute_v4_measurement_artifacts(
        run_dir=str(run_path),
        question_metadata=question_metadata,
        selection_rows=selection_rows,
        prompt_logits_fn=backend["prompt_logits_fn"],
        tokenizer=backend["tokenizer"],
        token_counter=token_counter,
        seed=selection_config.seed,
        token_delta_max=config.nldd.corruption_token_delta_max,
        retry_limit=config.nldd.corruption_retry_limit,
        ld_epsilon=config.nldd.ld_epsilon,
        integer_perturbation_range=tuple(config.nldd.integer_perturbation_range),
        float_perturbation_range=tuple(config.nldd.float_perturbation_range),
        enable_tier3_semantic_flip=config.nldd.enable_tier3_semantic_flip,
    )

    print(f"runtime_device_requested: {backend['runtime_selection'].requested_device}")
    print(f"runtime_device_resolved: {backend['runtime_selection'].resolved_device}")
    print(f"trace_selection_source: {selection_source}")
    print(f"selected_traces: {len(selection_rows)}")
    print(f"s_value: {artifacts['s_value']}")
    print(f"s_calibration_path: {artifacts['s_calibration_path']}")
    print(f"nldd_full_path: {artifacts['nldd_full_path']}")
    print(f"corruption_summary_path: {artifacts['corruption_summary_path']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Canonical Stage C/D run directory.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used to resolve model and analysis settings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for deterministic trace/corruption ordering.",
    )
    return parser.parse_args()


def load_measurement_backend(config: ExperimentConfig) -> dict[str, Any]:
    """Load the model, tokenizer, and runtime objects used for prompt scoring."""

    import torch
    from transformers import AutoModelForCausalLM

    runtime_selection = select_runtime_device(torch)
    local_model_path, downloaded = ensure_model_available(
        model_name=config.model.name,
        cache_dir=config.model.hf_cache,
    )
    tokenizer = _load_tokenizer_with_fallback(
        tokenizer_path=local_model_path,
        cache_dir=config.model.hf_cache,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=_resolve_torch_dtype(config.model.dtype, torch),
        cache_dir=config.model.hf_cache,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device(runtime_selection.resolved_device)
    model.to(device)
    model.eval()

    print(f"model_cache_hit: {not downloaded}")
    print(f"resolved_model_path: {local_model_path}")
    prompt_logits_fn = build_prompt_logit_fn(
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_module=torch,
    )
    return {
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
        "prompt_logits_fn": prompt_logits_fn,
        "runtime_selection": runtime_selection,
    }


if __name__ == "__main__":
    main()
