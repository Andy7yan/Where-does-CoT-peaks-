"""Reusable backend loading for analysis-side model evaluation."""

from __future__ import annotations

from typing import Any

from src.analysis_phase1.analysis import build_prompt_hidden_state_fn, build_trace_trajectory_fn
from src.analysis_phase1.nldd_measurement import build_prompt_logit_fn
from src.common.runtime_env import select_runtime_device
from src.common.settings import ExperimentConfig
from src.data_phase1.generation import _load_tokenizer_with_fallback, _resolve_torch_dtype, ensure_model_available


def load_analysis_backend(config: ExperimentConfig) -> dict[str, Any]:
    """Load the model, tokenizer, and callable backends used by analysis."""

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


__all__ = ["load_analysis_backend"]
