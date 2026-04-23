"""Experiment settings loading for peak-CoT."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class ExperimentMetadataConfig:
    """Top-level experiment metadata."""

    run_id: str
    seed: int


@dataclass
class DatasetConfig:
    """Dataset selection and ranking controls."""

    name: str
    hf_config: str | None
    split: str
    order_hash_seed: int


@dataclass
class ModelConfig:
    """Model and cache configuration."""

    name: str
    dtype: str
    hf_cache: str


@dataclass
class GenerationConfig:
    """ICL-driven generation settings."""

    num_icl_groups: int | None
    samples_per_group: int | None
    temperature: float | None
    icl_group_prompt_ids: list[str]
    icl_group_sample_counts: dict[str, int]
    max_new_tokens: int | None


@dataclass
class StepSegmentationConfig:
    """Rules for turning a completion into reasoning steps."""

    method: str
    answer_markers: list[str]


@dataclass
class AnswerExtractionConfig:
    """Rules for extracting numeric answers from completions."""

    numeric_tolerance: float


@dataclass
class NLDDConfig:
    """NLDD measurement configuration."""

    corruption_type: str
    integer_perturbation_range: list[int]
    float_perturbation_range: list[float]
    enable_tier3_semantic_flip: bool
    corruption_token_delta_max: int
    corruption_retry_limit: int
    perplexity_filter_enabled: bool
    perplexity_ratio_threshold: float | None
    ld_epsilon: float
    horizon_definition: str


@dataclass
class PilotConfig:
    """Pilot-run configuration."""

    num_questions: int


@dataclass
class TASConfig:
    """Trajectory analysis settings."""

    layer: str
    plateau_threshold: float | None


@dataclass
class AnalysisConfig:
    """Analysis-stage settings."""

    min_bin_size: int | None
    min_nldd_length: int | None
    hard_accuracy_threshold: float | None
    easy_accuracy_threshold: float | None
    num_length_bins: int | None
    length_bin_mode: str | None
    target_traces_per_cell: int | None
    target_traces_near_lstar: int | None
    per_question_trace_cap: int | None
    primary_lstar_window: int | None
    fallback_lstar_window: int | None
    min_near_lstar_traces: int | None
    min_cell_size: int | None
    num_normalized_bins: int | None
    min_bin_coverage_ratio: float | None
    num_full_analysis_questions: int | None
    max_extraction_fail_rate: float | None
    per_question_lcurve_min_bin_size: int | None
    per_question_min_retained_traces: int | None
    per_question_max_retained_traces: int | None
    per_question_lstar_smoothing_window: int | None
    per_question_min_lcurve_bins: int | None
    per_question_min_kstar_bins: int | None
    prompt_batch_size: int | None
    hidden_state_batch_size: int | None


@dataclass
class ExperimentConfig:
    """Typed representation of the Stage 1 YAML configuration."""

    experiment: ExperimentMetadataConfig
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    step_segmentation: StepSegmentationConfig
    answer_extraction: AnswerExtractionConfig
    nldd: NLDDConfig
    pilot: PilotConfig
    tas: TASConfig
    analysis: AnalysisConfig

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load YAML, expand environment variables, and build dataclasses."""

        data = load_settings(path)
        experiment = _require_mapping(data, "experiment")
        dataset = _require_mapping(data, "dataset")
        model = _require_mapping(data, "model")
        generation = _require_mapping(data, "generation")
        icl_group_prompt_ids = _parse_icl_group_prompt_ids(generation)
        step_segmentation = _require_mapping(data, "step_segmentation")
        answer_extraction = _require_mapping(data, "answer_extraction")
        nldd = _require_mapping(data, "nldd")
        pilot = _require_mapping(data, "pilot")
        tas = _require_mapping(data, "tas")
        analysis = _require_mapping(data, "analysis")
        integer_perturbation_range = _require_int_list(
            nldd,
            "integer_perturbation_range",
            expected_length=2,
        )
        _validate_integer_perturbation_range(integer_perturbation_range)
        float_perturbation_range = _require_float_list(
            nldd,
            "float_perturbation_range",
        )
        _validate_float_perturbation_range(float_perturbation_range)

        return cls(
            experiment=ExperimentMetadataConfig(
                run_id=_require_string(experiment, "run_id"),
                seed=_require_int(experiment, "seed"),
            ),
            dataset=DatasetConfig(
                name=_require_string(dataset, "name"),
                hf_config=_optional_string(dataset, "hf_config"),
                split=_require_string(dataset, "split"),
                order_hash_seed=_require_int(dataset, "order_hash_seed"),
            ),
            model=ModelConfig(
                name=_require_string(model, "name"),
                dtype=_require_string(model, "dtype"),
                hf_cache=_require_string(model, "hf_cache"),
            ),
            generation=GenerationConfig(
                num_icl_groups=_optional_int(generation, "num_icl_groups"),
                samples_per_group=_optional_int(generation, "samples_per_group"),
                temperature=_optional_float(generation, "temperature"),
                icl_group_prompt_ids=icl_group_prompt_ids,
                icl_group_sample_counts=_parse_icl_group_sample_counts(generation),
                max_new_tokens=_optional_int(generation, "max_new_tokens"),
            ),
            step_segmentation=StepSegmentationConfig(
                method=_require_string(step_segmentation, "method"),
                answer_markers=_require_string_list(step_segmentation, "answer_markers"),
            ),
            answer_extraction=AnswerExtractionConfig(
                numeric_tolerance=_require_float(answer_extraction, "numeric_tolerance"),
            ),
            nldd=NLDDConfig(
                corruption_type=_require_string(nldd, "corruption_type"),
                integer_perturbation_range=integer_perturbation_range,
                float_perturbation_range=float_perturbation_range,
                enable_tier3_semantic_flip=_require_bool(
                    nldd,
                    "enable_tier3_semantic_flip",
                ),
                corruption_token_delta_max=_require_int(
                    nldd,
                    "corruption_token_delta_max",
                ),
                corruption_retry_limit=_require_int(nldd, "corruption_retry_limit"),
                perplexity_filter_enabled=_require_bool(
                    nldd,
                    "perplexity_filter_enabled",
                ),
                perplexity_ratio_threshold=_optional_float(
                    nldd,
                    "perplexity_ratio_threshold",
                ),
                ld_epsilon=_require_float(nldd, "ld_epsilon"),
                horizon_definition=_require_string(nldd, "horizon_definition"),
            ),
            pilot=PilotConfig(
                num_questions=_require_int(pilot, "num_questions"),
            ),
            tas=TASConfig(
                layer=_require_string(tas, "layer"),
                plateau_threshold=_optional_float(tas, "plateau_threshold"),
            ),
            analysis=AnalysisConfig(
                min_bin_size=_optional_int(analysis, "min_bin_size"),
                min_nldd_length=_optional_int(analysis, "min_nldd_length"),
                hard_accuracy_threshold=_optional_float(
                    analysis,
                    "hard_accuracy_threshold",
                ),
                easy_accuracy_threshold=_optional_float(
                    analysis,
                    "easy_accuracy_threshold",
                ),
                num_length_bins=_optional_int(analysis, "num_length_bins"),
                length_bin_mode=_optional_string(analysis, "length_bin_mode"),
                target_traces_per_cell=_optional_int(
                    analysis,
                    "target_traces_per_cell",
                ),
                target_traces_near_lstar=_optional_int(
                    analysis,
                    "target_traces_near_lstar",
                ),
                per_question_trace_cap=_optional_int(
                    analysis,
                    "per_question_trace_cap",
                ),
                primary_lstar_window=_optional_int(
                    analysis,
                    "primary_lstar_window",
                ),
                fallback_lstar_window=_optional_int(
                    analysis,
                    "fallback_lstar_window",
                ),
                min_near_lstar_traces=_optional_int(
                    analysis,
                    "min_near_lstar_traces",
                ),
                min_cell_size=_optional_int(analysis, "min_cell_size"),
                num_normalized_bins=_optional_int(
                    analysis,
                    "num_normalized_bins",
                ),
                min_bin_coverage_ratio=_optional_float(
                    analysis,
                    "min_bin_coverage_ratio",
                ),
                num_full_analysis_questions=_optional_int(
                    analysis,
                    "num_full_analysis_questions",
                ),
                max_extraction_fail_rate=_optional_float(
                    analysis,
                    "max_extraction_fail_rate",
                ),
                per_question_lcurve_min_bin_size=_optional_int(
                    analysis,
                    "per_question_lcurve_min_bin_size",
                ),
                per_question_min_retained_traces=_optional_int(
                    analysis,
                    "per_question_min_retained_traces",
                ),
                per_question_max_retained_traces=_optional_int(
                    analysis,
                    "per_question_max_retained_traces",
                ),
                per_question_lstar_smoothing_window=_optional_int(
                    analysis,
                    "per_question_lstar_smoothing_window",
                ),
                per_question_min_lcurve_bins=_optional_int(
                    analysis,
                    "per_question_min_lcurve_bins",
                ),
                per_question_min_kstar_bins=_optional_int(
                    analysis,
                    "per_question_min_kstar_bins",
                ),
                prompt_batch_size=_optional_int(
                    analysis,
                    "prompt_batch_size",
                ),
                hidden_state_batch_size=_optional_int(
                    analysis,
                    "hidden_state_batch_size",
                ),
            ),
        )


def load_settings(path: str) -> dict[str, Any]:
    """Read YAML, expand environment variables, and return the raw mapping."""

    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.is_dir():
        raise IsADirectoryError(
            f"Config path points to a directory, not a YAML file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping.")

    expanded = _expand_env_vars(data)
    if not isinstance(expanded, dict):
        raise TypeError("Expanded config root must be a mapping.")
    return expanded


def load_config(path: str) -> dict[str, Any]:
    """Backward-compatible alias for settings loading."""

    return load_settings(path)


def require_config_value(field_path: str, value: T | None) -> T:
    """Require a config value to be filled after the pilot gate."""

    if value is None:
        raise ValueError(f"{field_path} 需由 Pilot Run 确认后填写")
    return value


def _resolve_config_path(path: str) -> Path:
    raw_path = path.strip() if isinstance(path, str) else str(path)
    if not raw_path:
        return Path("configs/stage1.yaml")

    config_path = Path(raw_path)
    if config_path.is_dir():
        for candidate in (
            config_path / "configs" / "stage1.yaml",
            config_path / "stage1.yaml",
        ):
            if candidate.exists() and candidate.is_file():
                return candidate
    return config_path


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key)
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return section


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Config field '{key}' must be a string.")
    return value


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"Config field '{key}' must be a boolean.")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config field '{key}' must be an integer.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config field '{key}' must be an integer or null.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Config field '{key}' must be a string or null.")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    return _coerce_float(value, key, allow_null=False)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _coerce_float(value, key, allow_null=True)


def _parse_icl_group_prompt_ids(data: dict[str, Any]) -> list[str]:
    raw_groups = _require_icl_groups_mapping(data)
    return list(raw_groups)


def _parse_icl_group_sample_counts(data: dict[str, Any]) -> dict[str, int]:
    raw_groups = _require_icl_groups_mapping(data)

    sample_counts: dict[str, int] = {}
    for prompt_id, group_config in raw_groups.items():
        if "samples_per_group" in group_config:
            sample_counts[prompt_id] = _require_int(group_config, "samples_per_group")
    return sample_counts


def _require_icl_groups_mapping(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_groups = data.get("icl_groups")
    if raw_groups is None:
        return {}
    if not isinstance(raw_groups, dict):
        raise TypeError("Config field 'icl_groups' must be a mapping.")

    normalized: dict[str, dict[str, Any]] = {}
    for prompt_id, group_config in raw_groups.items():
        if not isinstance(prompt_id, str):
            raise TypeError("Config field 'icl_groups' must use string prompt ids.")
        if not isinstance(group_config, dict):
            raise TypeError(
                f"Config field 'icl_groups.{prompt_id}' must be a mapping."
            )
        normalized[prompt_id] = group_config
    return normalized


def _require_float_list(
    data: dict[str, Any],
    key: str,
    *,
    expected_length: int | None = None,
) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Config field '{key}' must be a list.")
    converted: list[float] = []
    for item in value:
        converted.append(_coerce_float(item, key, allow_null=False))
    if expected_length is not None and len(converted) != expected_length:
        raise TypeError(
            f"Config field '{key}' must contain exactly {expected_length} floats."
        )
    return converted


def _require_int_list(
    data: dict[str, Any],
    key: str,
    *,
    expected_length: int | None = None,
) -> list[int]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Config field '{key}' must be a list.")
    converted: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"Config field '{key}' must contain only integers.")
        converted.append(item)
    if expected_length is not None and len(converted) != expected_length:
        raise TypeError(
            f"Config field '{key}' must contain exactly {expected_length} integers."
        )
    return converted


def _optional_float_list(
    data: dict[str, Any],
    key: str,
    *,
    expected_length: int | None = None,
) -> list[float] | None:
    value = data.get(key)
    if value is None:
        return None
    return _require_float_list(
        data,
        key,
        expected_length=expected_length,
    )


def _require_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"Config field '{key}' must be a list.")
    converted: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"Config field '{key}' must contain only strings.")
        converted.append(item)
    return converted


def _coerce_float(value: Any, key: str, allow_null: bool) -> float:
    if value is None and allow_null:
        raise TypeError(f"Config field '{key}' must be a float or null.")
    if isinstance(value, bool):
        raise TypeError(f"Config field '{key}' must be a float.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(f"Config field '{key}' must be a float.") from exc
    if allow_null:
        raise TypeError(f"Config field '{key}' must be a float or null.")
    raise TypeError(f"Config field '{key}' must be a float.")


def _validate_integer_perturbation_range(values: list[int]) -> None:
    low, high = values
    if not (low < 0 < high):
        raise TypeError(
            "Config field 'integer_perturbation_range' must satisfy low < 0 < high."
        )


def _validate_float_perturbation_range(values: list[float]) -> None:
    if len(values) == 2:
        low, high = values
        if not (0.0 < low < high):
            raise TypeError(
                "Config field 'float_perturbation_range' must satisfy "
                "0.0 < low < high."
            )
        return

    if len(values) == 4:
        low_a, high_a, low_b, high_b = values
        if not (0.0 < low_a < high_a < 1.0 < low_b < high_b):
            raise TypeError(
                "Config field 'float_perturbation_range' must satisfy "
                "0.0 < low_a < high_a < 1.0 < low_b < high_b when two "
                "disjoint intervals are provided."
            )
        return

    raise TypeError(
        "Config field 'float_perturbation_range' must contain either 2 floats "
        "(single interval) or 4 floats (two disjoint intervals)."
    )


__all__ = [
    "AnalysisConfig",
    "AnswerExtractionConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ExperimentMetadataConfig",
    "GenerationConfig",
    "ModelConfig",
    "NLDDConfig",
    "PilotConfig",
    "StepSegmentationConfig",
    "TASConfig",
    "load_config",
    "load_settings",
    "require_config_value",
]
