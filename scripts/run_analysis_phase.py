"""Run the analysis workflow on the canonical difficulty/sample layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase.analysis import (
    build_prompt_hidden_state_fn,
    build_trace_trajectory_fn,
    run_analysis,
)
from src.analysis_phase.nldd_measurement import build_prompt_logit_fn
from src.common.runtime_env import select_runtime_device
from src.common.settings import ExperimentConfig
from src.data_phase1.generation import _load_tokenizer_with_fallback, _resolve_torch_dtype, ensure_model_available


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
    print(f"analysis_dir: {artifacts['analysis_dir']}")
    print(f"sample_count: {artifacts['sample_count']}")
    print(f"s_value: {artifacts['s_value']}")
    print(f"accuracy_by_length_path: {artifacts['accuracy_by_length_path']}")
    print(f"nldd_per_trace_path: {artifacts['nldd_per_trace_path']}")
    print(f"tas_per_trace_path: {artifacts['tas_per_trace_path']}")


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


def load_analysis_backend(config: ExperimentConfig) -> dict[str, Any]:
    """Load the model, tokenizer, and callable backends used by the analysis flow."""

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
    prompt_hidden_state_fn = build_prompt_hidden_state_fn(
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_module=torch,
        layer=config.tas.layer,
    )
    return {
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
        "prompt_logits_fn": prompt_logits_fn,
        "trace_trajectory_fn": build_trace_trajectory_fn(
            prompt_hidden_state_fn=prompt_hidden_state_fn,
        ),
        "runtime_selection": runtime_selection,
    }


if __name__ == "__main__":
    main()
