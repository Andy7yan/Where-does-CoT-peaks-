"""Configuration loading utilities for peak-CoT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentMetadataConfig:
    """Top-level experiment metadata."""

    name: str
    seed: int


@dataclass
class DatasetConfig:
    """Dataset selection and subset controls."""

    name: str
    split: str
    subset_size: int
    subset_hash_seed: int


@dataclass
class ModelConfig:
    """Model and cache configuration."""

    name: str
    dtype: str
    hf_cache: str


@dataclass
class GenerationConfig:
    """Length-controlled generation settings."""

    length_grid: list[int]
    samples_per_length: int
    temperature: float
    max_new_tokens: int


@dataclass
class StepSegmentationConfig:
    """Rules for turning a completion into reasoning steps."""

    method: str
    strip_whitespace: bool
    drop_empty: bool
    answer_markers: list[str]


@dataclass
class AnswerExtractionConfig:
    """Rules for extracting numeric answers from completions."""

    method: str
    numeric_tolerance: float


@dataclass
class NLDDConfig:
    """NLDD measurement configuration."""

    corruption_type: str
    perturbation_range: list[float]
    confidence_metric: str
    normalization: str
    horizon_definition: str


@dataclass
class OutputConfig:
    """Output directory configuration."""

    base_dir: str


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
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load YAML, expand environment variables, and build dataclasses."""

        data = load_config(path)
        generation = _require_mapping(data, "generation")
        answer_extraction = _require_mapping(data, "answer_extraction")
        nldd = _require_mapping(data, "nldd")
        return cls(
            experiment=ExperimentMetadataConfig(**_require_mapping(data, "experiment")),
            dataset=DatasetConfig(**_require_mapping(data, "dataset")),
            model=ModelConfig(**_require_mapping(data, "model")),
            generation=GenerationConfig(
                length_grid=[int(item) for item in generation["length_grid"]],
                samples_per_length=int(generation["samples_per_length"]),
                temperature=float(generation["temperature"]),
                max_new_tokens=int(generation["max_new_tokens"]),
            ),
            step_segmentation=StepSegmentationConfig(
                **_require_mapping(data, "step_segmentation")
            ),
            answer_extraction=AnswerExtractionConfig(
                method=str(answer_extraction["method"]),
                numeric_tolerance=float(answer_extraction["numeric_tolerance"]),
            ),
            nldd=NLDDConfig(
                corruption_type=str(nldd["corruption_type"]),
                perturbation_range=[float(item) for item in nldd["perturbation_range"]],
                confidence_metric=str(nldd["confidence_metric"]),
                normalization=str(nldd["normalization"]),
                horizon_definition=str(nldd["horizon_definition"]),
            ),
            output=OutputConfig(**_require_mapping(data, "output")),
        )


def load_config(path: str) -> dict[str, Any]:
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
