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
    """Dataset selection and subset controls."""

    name: str
    split: str
    subset_size: int | None
    subset_hash_seed: int


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
    icl_group_temperatures: dict[str, float]
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
    integer_perturbation: str
    float_perturbation_range: list[float]
    enable_tier3_semantic_flip: bool
    corruption_token_delta_max: int
    corruption_retry_limit: int
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
    num_full_analysis_questions: int | None
    num_spot_checks: int | None
    max_extraction_fail_rate: float | None


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
        float_perturbation_range = _require_float_list(
            nldd,
            "float_perturbation_range",
            expected_length=4,
        )
        _validate_float_perturbation_range(float_perturbation_range)

        return cls(
            experiment=ExperimentMetadataConfig(
                run_id=_require_string(experiment, "run_id"),
                seed=_require_int(experiment, "seed"),
            ),
            dataset=DatasetConfig(
                name=_require_string(dataset, "name"),
                split=_require_string(dataset, "split"),
                subset_size=_optional_int(dataset, "subset_size"),
                subset_hash_seed=_require_int(dataset, "subset_hash_seed"),
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
                icl_group_temperatures=_parse_icl_group_temperatures(generation),
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
                integer_perturbation=_require_string(nldd, "integer_perturbation"),
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
                num_full_analysis_questions=_optional_int(
                    analysis,
                    "num_full_analysis_questions",
                ),
                num_spot_checks=_optional_int(analysis, "num_spot_checks"),
                max_extraction_fail_rate=_optional_float(
                    analysis,
                    "max_extraction_fail_rate",
                ),
            ),
        )


def load_settings(path: str) -> dict[str, Any]:
    """Read YAML, expand environment variables, and return the raw mapping."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

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


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    return _coerce_float(value, key, allow_null=False)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _coerce_float(value, key, allow_null=True)


def _parse_icl_group_temperatures(data: dict[str, Any]) -> dict[str, float]:
    raw_groups = _require_icl_groups_mapping(data)

    temperatures: dict[str, float] = {}
    for prompt_id, group_config in raw_groups.items():
        if "temperature" in group_config:
            temperatures[prompt_id] = _require_float(group_config, "temperature")
    return temperatures


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


def _validate_float_perturbation_range(values: list[float]) -> None:
    low_min, low_max, high_min, high_max = values
    if not (low_min < low_max < high_min < high_max):
        raise TypeError(
            "Config field 'float_perturbation_range' must satisfy "
            "low_min < low_max < high_min < high_max."
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
