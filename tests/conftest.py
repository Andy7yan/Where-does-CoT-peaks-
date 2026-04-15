"""Shared pytest fixtures for peak-CoT."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_gsm8k() -> list[dict]:
    """Load the bundled sample records."""

    sample_path = Path(__file__).resolve().parents[1] / "data" / "gsm8k_sample.json"
    return json.loads(sample_path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_question(sample_gsm8k: list[dict]) -> dict:
    """Return the first bundled sample question."""

    return sample_gsm8k[0]
