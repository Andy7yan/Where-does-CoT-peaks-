"""ICL-driven trace generation utilities for Stage 1."""

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from src.data_phase1.prompting import build_generation_messages
from src.common.reasoning import extract_answer, judge, segment_steps
from src.common.runtime_env import select_runtime_device


@dataclass(frozen=True)
class GenerationOutput:
    """A decoded completion and its generated token count."""

    raw_completion: str
    token_count: int


TRACE_SCHEMA_VERSION = "stage1_trace_v2"
DEFAULT_GENERATION_BATCH_SIZE = 4


def ensure_model_available(
    model_name: str,
    cache_dir: str,
    hf_token: str | None = None,
) -> tuple[str, bool]:
    """Ensure that a Hugging Face model snapshot exists locally."""

    from huggingface_hub import snapshot_download

    resolved_token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    needs_refresh = False

    try:
        local_model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_path),
            local_files_only=True,
            token=resolved_token,
        )
        if _snapshot_has_required_files(local_model_path):
            return local_model_path, False
        print(
            "warning: cached model snapshot looks incomplete; forcing a fresh download",
            local_model_path,
        )
        needs_refresh = True
    except Exception as local_exc:
        if not resolved_token:
            raise RuntimeError(
                f"Model '{model_name}' is not present in cache '{cache_dir}', and HF_TOKEN is not set. "
                "Set HF_TOKEN and ensure you have accepted the model license before retrying."
            ) from local_exc

    try:
        local_model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_path),
            token=resolved_token,
            force_download=needs_refresh,
        )
        if not _snapshot_has_required_files(local_model_path):
            raise RuntimeError(
                f"Downloaded snapshot for '{model_name}' is still incomplete at '{local_model_path}'. "
                "Clear the model cache directory and retry the download."
            )
        return local_model_path, True
    except Exception as download_exc:
        raise RuntimeError(
            f"Failed to download model '{model_name}' into cache '{cache_dir}'. "
            "Check HF_TOKEN, network access, and gated-model permissions."
        ) from download_exc


class LLMGenerator:
    """Manage the tokenizer and model lifecycle for length-controlled generation."""

    def __init__(self, model_name: str, dtype: str = "float16", cache_dir: str | None = None):
        """Load the tokenizer and model, logging key runtime diagnostics."""

        import torch
        from transformers import AutoModelForCausalLM

        self.model_name = model_name
        self.dtype_name = dtype
        self.cache_dir = cache_dir or os.getenv(
            "HF_HUB_CACHE",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
        self.runtime_device = select_runtime_device(torch)
        self.device = torch.device(self.runtime_device.resolved_device)
        self._input_diagnostics_logged = False
        self._generation_configs_logged: set[tuple[bool, float, int, int | None, str]] = set()

        load_start = time.perf_counter()
        local_model_path, downloaded = ensure_model_available(
            model_name=model_name,
            cache_dir=self.cache_dir,
        )

        torch_dtype = _resolve_torch_dtype(dtype, torch)
        self.tokenizer = _load_tokenizer_with_fallback(
            tokenizer_path=local_model_path,
            cache_dir=self.cache_dir,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            dtype=torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        load_elapsed = time.perf_counter() - load_start
        model_device = next(self.model.parameters()).device
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

        print(
            "model_init:",
            {
                "model_name": self.model_name,
                "resolved_device": self.runtime_device.resolved_device,
                "gpu_name": self.runtime_device.gpu_name,
                "dtype": self.dtype_name,
                "parameter_count": parameter_count,
                "load_seconds": round(load_elapsed, 3),
                "cache_hit": not downloaded,
            },
        )
        _debug_log(
            "model_init_details="
            + repr(
                {
                    "requested_device": self.runtime_device.requested_device,
                    "device_resolution_reason": self.runtime_device.reason,
                    "gpu_compute_capability": self.runtime_device.gpu_compute_capability,
                    "torch_supported_cuda_arches": list(
                        self.runtime_device.supported_cuda_arches
                    ),
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                    "model_device": str(model_device),
                    "downloaded_model": downloaded,
                    "local_model_path": local_model_path,
                    "tokenizer_class": self.tokenizer.__class__.__name__,
                }
            )
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        """Run a single autoregressive generation from chat messages."""

        return self.generate_batch(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list[GenerationOutput]:
        """Run batched autoregressive generation from one or more chat prompts."""

        import torch

        if not messages_batch:
            return []

        if len(messages_batch) == 1:
            model_inputs = self._prepare_model_inputs(messages_batch[0])
        else:
            model_inputs = self._prepare_batched_model_inputs(messages_batch)
        if "input_ids" not in model_inputs:
            raise RuntimeError("Tokenizer chat template did not return input_ids.")
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(
                model_inputs["input_ids"],
                device=self.device,
            )

        input_ids = model_inputs["input_ids"]
        _debug_log(f"prepared_input_keys={list(model_inputs.keys())}")
        _debug_log(f"prepared_input_ids_type={type(input_ids).__name__}")
        _debug_log(f"prepared_input_ids_shape={getattr(input_ids, 'shape', None)}")

        if not self._input_diagnostics_logged:
            _debug_log(f"input_device={input_ids.device}")
            _debug_log(f"batch_shape={tuple(input_ids.shape)}")
            self._input_diagnostics_logged = True

        generation_config = self._build_generation_config(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                generation_config=generation_config,
            )

        input_width = input_ids.shape[-1]
        pad_token_id = generation_config.pad_token_id
        generations: list[GenerationOutput] = []
        for row in outputs:
            generated_tokens = row[input_width:]
            decoded_tokens = (
                generated_tokens[generated_tokens != pad_token_id]
                if pad_token_id is not None
                else generated_tokens
            )
            raw_completion = self.tokenizer.decode(
                decoded_tokens,
                skip_special_tokens=True,
            )
            generations.append(
                GenerationOutput(
                    raw_completion=raw_completion,
                    token_count=int(decoded_tokens.shape[-1]),
                )
            )
        return generations

    def _prepare_model_inputs(self, messages: list[dict]) -> dict[str, Any]:
        """Prepare model inputs from chat messages with a robust backend fallback."""

        try:
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            _debug_log(f"chat_template_tokenize_true_type={type(model_inputs).__name__}")
            _debug_log(f"chat_template_tokenize_true_repr={_short_repr(model_inputs)}")
            return _move_model_inputs_to_device(model_inputs, self.device)
        except Exception as exc:
            print(
                "warning: tokenizer.apply_chat_template(tokenize=True) returned an unsupported format; "
                "falling back to string template + tokenizer(...)",
            )
            print("chat_template_error_type=", type(exc).__name__)
            _debug_log(f"chat_template_error_repr={exc!r}")

            rendered_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            _debug_log(f"chat_template_tokenize_false_type={type(rendered_prompt).__name__}")
            _debug_log(f"chat_template_tokenize_false_preview={_short_repr(rendered_prompt)}")
            if not isinstance(rendered_prompt, str):
                raise RuntimeError(
                    "Tokenizer chat template fallback failed to produce a string prompt."
                ) from exc

            tokenized = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            _debug_log(f"fallback_tokenizer_call_type={type(tokenized).__name__}")
            _debug_log(f"fallback_tokenizer_call_repr={_short_repr(tokenized)}")
            return _move_model_inputs_to_device(tokenized, self.device)

    def _prepare_batched_model_inputs(
        self,
        messages_batch: list[list[dict]],
    ) -> dict[str, Any]:
        """Prepare left-padded model inputs for a batch of chat prompts."""

        rendered_prompts = [
            self._render_prompt_from_messages(messages)
            for messages in messages_batch
        ]
        tokenized = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        _debug_log(f"batched_tokenizer_call_type={type(tokenized).__name__}")
        _debug_log(f"batched_tokenizer_call_repr={_short_repr(tokenized)}")
        return _move_model_inputs_to_device(tokenized, self.device)

    def _render_prompt_from_messages(self, messages: list[dict]) -> str:
        """Render chat messages to a prompt string for batched tokenization."""

        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered_prompt, str):
            raise RuntimeError("Tokenizer chat template did not return a prompt string.")
        return rendered_prompt

    def _build_generation_config(
        self,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> Any:
        """Build and log the effective generation config for this call."""

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.do_sample = temperature > 0
        generation_config.temperature = temperature
        config_key = (
            bool(generation_config.do_sample),
            float(generation_config.temperature),
            int(generation_config.max_new_tokens),
            generation_config.pad_token_id,
            repr(generation_config.eos_token_id),
        )
        if config_key not in self._generation_configs_logged:
            print(
                "generation_config:",
                {
                    "do_sample": generation_config.do_sample,
                    "temperature": generation_config.temperature,
                    "max_new_tokens": generation_config.max_new_tokens,
                    "pad_token_id": generation_config.pad_token_id,
                    "eos_token_id": generation_config.eos_token_id,
                },
            )
            self._generation_configs_logged.add(config_key)
        return generation_config


def generate_traces_for_question(
    generator: LLMGenerator,
    question_id: str,
    question_text: str,
    gold_answer: float,
    prompt_templates: list[dict[str, Any]],
    samples_per_group: int | None,
    max_new_tokens: int,
    temperature: float | None = None,
    prompt_sample_counts: Mapping[str, int] | None = None,
    batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
) -> list[dict]:
    """Generate all Stage 1 traces for a single question across ICL prompt groups."""

    traces: list[dict[str, Any]] = []
    for prompt_template in prompt_templates:
        prompt_id = prompt_template.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise TypeError("prompt_template['prompt_id'] must be a string.")
        _debug_log(f"prompt_id={prompt_id}")
        messages = build_generation_messages(
            question=question_text,
            prompt_template=prompt_template,
        )
        if temperature is None:
            raise ValueError("No global generation temperature configured.")
        effective_samples_per_group = _resolve_prompt_sample_count(
            prompt_id=prompt_id,
            samples_per_group=samples_per_group,
            prompt_sample_counts=prompt_sample_counts,
        )
        sample_idx = 1
        while sample_idx <= effective_samples_per_group:
            batch_sample_end = min(
                sample_idx + max(batch_size, 1) - 1,
                effective_samples_per_group,
            )
            batch_sample_indices = list(range(sample_idx, batch_sample_end + 1))
            generation_start = time.perf_counter()
            _debug_log(
                "samples="
                f"{batch_sample_indices[0]}-{batch_sample_indices[-1]}/{effective_samples_per_group} "
                f"generating prompt_id={prompt_id} temperature={temperature}"
            )
            if hasattr(generator, "generate_batch"):
                generations = generator.generate_batch(
                    messages_batch=[messages for _ in batch_sample_indices],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            else:
                generations = [
                    generator.generate(
                        messages=messages,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
                    for _ in batch_sample_indices
                ]
            if len(generations) != len(batch_sample_indices):
                raise RuntimeError("Generator returned a mismatched number of batched outputs.")
            batch_seconds = time.perf_counter() - generation_start
            _debug_log(
                "samples="
                f"{batch_sample_indices[0]}-{batch_sample_indices[-1]}/{effective_samples_per_group} done "
                f"prompt_id={prompt_id} seconds={batch_seconds:.2f} "
                f"tokens={[generation.token_count for generation in generations]}"
            )
            for current_sample_idx, generation in zip(batch_sample_indices, generations):
                segmentation = segment_steps(generation.raw_completion)
                extraction = extract_answer(generation.raw_completion)
                traces.append(
                    {
                        "trace_id": f"{question_id}_{prompt_id}_{current_sample_idx}",
                        "question_id": question_id,
                        "question_text": question_text,
                        "gold_answer": gold_answer,
                        "prompt_id": prompt_id,
                        "raw_completion": generation.raw_completion,
                        "steps": segmentation.steps,
                        "actual_num_steps": segmentation.num_steps,
                        "final_answer_line": segmentation.final_answer_line,
                        "extracted_answer": extraction.value,
                        "is_correct": judge(extraction.value, gold_answer),
                        "extraction_failed": extraction.extraction_failed,
                        "token_count": generation.token_count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            sample_idx = batch_sample_end + 1

    return traces


def _resolve_prompt_sample_count(
    *,
    prompt_id: str,
    samples_per_group: int | None,
    prompt_sample_counts: Mapping[str, int] | None,
) -> int:
    """Resolve the effective sample count for a prompt group."""

    if prompt_sample_counts is not None and prompt_id in prompt_sample_counts:
        return int(prompt_sample_counts[prompt_id])
    if samples_per_group is None:
        raise ValueError(
            f"No sample count configured for prompt group '{prompt_id}'."
        )
    return samples_per_group


def write_run_metadata(output_dir: str, run_metadata: dict[str, Any]) -> str:
    """Write run-level metadata into run_meta.json."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    meta_path = output_path / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return str(meta_path)


def validate_output_dir_schema(output_dir: str, expected_schema_version: str) -> None:
    """Abort when an output directory mixes incompatible trace schemas."""

    output_path = Path(output_dir)
    traces_path = output_path / "traces.jsonl"
    meta_path = output_path / "run_meta.json"

    if traces_path.exists() and not meta_path.exists():
        raise RuntimeError(
            "Output directory contains traces.jsonl but is missing run_meta.json. "
            "Use a fresh output directory for Stage 1 trace schema v2."
        )

    if not meta_path.exists():
        return

    with meta_path.open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)

    if run_metadata.get("schema_version") != expected_schema_version:
        raise RuntimeError(
            "Output directory contains an incompatible run_meta.json schema_version. "
            "Use a fresh output directory for Stage 1 trace schema v2."
        )


def append_traces_to_jsonl(traces: list[dict], output_path: str) -> None:
    """Append trace dictionaries to a JSONL file."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_trace_ids(output_path: str) -> set[str]:
    """Load the set of existing trace ids from a JSONL file."""

    output_file = Path(output_path)
    if not output_file.exists():
        return set()

    trace_ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            trace_id = payload.get("trace_id")
            if isinstance(trace_id, str):
                trace_ids.add(trace_id)
    return trace_ids


def _resolve_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    normalized = dtype_name.lower()
    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return dtype_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def _move_model_inputs_to_device(model_inputs: Any, device: Any) -> dict[str, Any]:
    """Move chat-template outputs to the target device and normalize to a dict."""

    _debug_log(f"_move_model_inputs_to_device_input_type={type(model_inputs).__name__}")
    if _looks_like_tensor(model_inputs):
        _debug_log("_move_model_inputs_to_device_branch=tensor_like")
        return {"input_ids": model_inputs.to(device)}

    if isinstance(model_inputs, (list, tuple)):
        import torch

        _debug_log("_move_model_inputs_to_device_branch=list_or_tuple")
        return {"input_ids": torch.tensor([model_inputs], device=device)}

    if hasattr(model_inputs, "to"):
        _debug_log("_move_model_inputs_to_device_calling_to()")
        model_inputs = model_inputs.to(device)

    if isinstance(model_inputs, Mapping) or hasattr(model_inputs, "items"):
        _debug_log("_move_model_inputs_to_device_branch=mapping_like")
        moved: dict[str, Any] = {}
        for key, value in model_inputs.items():
            moved[key] = value.to(device) if _looks_like_tensor(value) else value
        return moved

    raise TypeError(
        "Tokenizer apply_chat_template returned an unsupported input type. "
        "Expected a BatchEncoding or dict-like object."
    )


def _looks_like_tensor(value: Any) -> bool:
    """Return whether a value behaves like a torch tensor for our purposes."""

    return hasattr(value, "shape") and hasattr(value, "to")


def _debug_log(message: str) -> None:
    """Print debug logs only when explicitly enabled."""

    if os.getenv("PEAK_COT_DEBUG") == "1":
        print(f"[debug] {message}")


def _short_repr(value: Any, limit: int = 400) -> str:
    """Return a truncated repr for debug logging."""

    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _snapshot_has_required_files(snapshot_path: str | os.PathLike[str]) -> bool:
    """Return whether a local Hugging Face snapshot looks usable."""

    path = Path(snapshot_path)
    required_any = [
        ("config.json",),
        ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"),
        ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"),
    ]
    return all(
        any((path / candidate).exists() for candidate in candidates)
        for candidates in required_any
    )


def _load_tokenizer_with_fallback(tokenizer_path: str, cache_dir: str):
    """Load a tokenizer, retrying with the slow backend when needed."""

    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=cache_dir,
                local_files_only=True,
                use_fast=False,
            )
        except Exception as slow_exc:
            raise RuntimeError(
                "Failed to load the tokenizer. "
                "This usually means the environment is missing tokenizer dependencies "
                "such as protobuf, sentencepiece, or tiktoken. "
                "Install them in the current environment and retry."
            ) from slow_exc
