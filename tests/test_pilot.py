"""Tests for the Stage D Pilot runner."""

import json
from pathlib import Path
import shutil
import uuid

import yaml

from src.generation import TRACE_SCHEMA_VERSION
from src.pilot import discover_prompt_templates, parse_pilot_overrides, run_pilot
from src.settings import load_settings


def test_parse_pilot_overrides_from_raw_settings(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "pilot-test")
    settings = load_settings("configs/stage1.yaml")

    pilot = parse_pilot_overrides(settings)

    assert pilot.num_questions == 50
    assert pilot.num_icl_groups == 3
    assert pilot.samples_per_group == 2
    assert pilot.temperature == 0.0
    assert pilot.max_new_tokens == 128
    assert pilot.max_extraction_fail_rate == 0.05


def test_discover_prompt_templates_validates_count() -> None:
    temp_dir = Path("tests") / f"_tmp_pilot_prompts_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "icl_short.yaml").write_text(
            'prompt_id: "icl_short"\nversion: 1\nsystem: "s"\nfew_shot: []\nuser_template: "{question}"\n',
            encoding="utf-8",
        )
        (temp_dir / "icl_medium.yaml").write_text(
            'prompt_id: "icl_medium"\nversion: 1\nsystem: "m"\nfew_shot: []\nuser_template: "{question}"\n',
            encoding="utf-8",
        )

        templates = discover_prompt_templates(str(temp_dir), expected_count=2)

        assert [template["prompt_id"] for template in templates] == [
            "icl_medium",
            "icl_short",
        ]

        try:
            discover_prompt_templates(str(temp_dir), expected_count=3)
        except ValueError as exc:
            assert "Expected 3 ICL prompt groups" in str(exc)
        else:
            raise AssertionError("discover_prompt_templates should reject mismatched counts")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_pilot_mock_writes_outputs_and_report(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setenv("RUN_NAME", "pilot-mock")

    temp_dir = Path("tests") / f"_tmp_pilot_run_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_dir / "stage1_pilot_test.yaml"
        output_dir = temp_dir / "pilot-output"

        base_config = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
        base_config["pilot"]["num_questions"] = 4
        base_config["pilot"]["num_icl_groups"] = 3
        base_config["pilot"]["samples_per_group"] = 2
        base_config["pilot"]["temperature"] = 0.0
        base_config["pilot"]["max_new_tokens"] = 64
        base_config["pilot"]["max_extraction_fail_rate"] = 0.05
        config_path.write_text(
            yaml.safe_dump(base_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        artifacts = run_pilot(
            config_path=str(config_path),
            output_dir=str(output_dir),
            mock=True,
            data_path="data/gsm8k_sample.json",
        )

        traces_path = Path(artifacts["pilot_traces_path"])
        meta_path = Path(artifacts["run_meta_path"])
        report_path = Path(artifacts["pilot_report_path"])

        assert traces_path.exists()
        assert meta_path.exists()
        assert report_path.exists()

        trace_lines = traces_path.read_text(encoding="utf-8").splitlines()
        assert len(trace_lines) == 24

        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert run_meta["schema_version"] == TRACE_SCHEMA_VERSION
        assert run_meta["mode"] == "mock"
        assert run_meta["num_questions"] == 4
        assert run_meta["num_icl_groups"] == 3
        assert run_meta["samples_per_group"] == 2

        report = report_path.read_text(encoding="utf-8")
        assert "# Pilot Report" in report
        assert "### A. ICL Length Guidance" in report
        assert "### B. Per-Length Sample Volume" in report
        assert "### C. Segmentation And Extraction" in report
        assert "### D. Corruption Feasibility" in report
        assert "### E. NLDD Smoke" in report
        assert "Deferred to Stage F by stage-boundary decision." in report
        for field_name in (
            "dataset.subset_size",
            "generation.num_icl_groups",
            "generation.samples_per_group",
            "generation.temperature",
            "generation.max_new_tokens",
            "analysis.min_bin_size",
            "analysis.max_extraction_fail_rate",
            "tas.plateau_threshold",
            "analysis.num_spot_checks",
        ):
            assert field_name in report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
