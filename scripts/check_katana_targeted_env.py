"""Preflight checks for the targeted relaxed rerun on Katana."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_targeted_rerun_manifest import DEFAULT_TARGET_QUESTION_CODES
from scripts.run_generation import discover_prompt_templates
from src.common.settings import ExperimentConfig
from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    load_per_question_manifest,
)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    scratch = Path(os.environ.get("SCRATCH", ""))
    source_run_dir = Path(args.source_run_dir) if args.source_run_dir else scratch / "runs" / args.source_run_name
    target_ids = {_normalize_question_id(value) for value in args.question_id}

    checks: list[dict[str, Any]] = []
    checks.extend(check_env_vars())
    checks.append(check_path("project_root", PROJECT_ROOT, expect_dir=True))
    checks.append(check_path("scratch", scratch, expect_dir=True))
    checks.append(check_path("source_run_dir", source_run_dir, expect_dir=True))
    checks.append(check_path("source_manifest", source_run_dir / PER_QUESTION_MANIFEST_FILENAME, expect_file=True))
    checks.extend(check_target_questions(source_run_dir / PER_QUESTION_MANIFEST_FILENAME, target_ids))
    checks.extend(check_prompts(args.prompts_dir, expected_count=config.generation.num_icl_groups or 4))
    checks.append(check_model_snapshot(config.model.name, config.model.hf_cache))

    report = {
        "status": "ok" if all(check["ok"] for check in checks) else "failed",
        "config": args.config,
        "source_run_dir": str(source_run_dir),
        "target_question_ids": sorted(target_ids),
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


def check_env_vars() -> list[dict[str, Any]]:
    required = ["SCRATCH", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"]
    checks = []
    for name in required:
        value = os.environ.get(name)
        checks.append(
            {
                "name": f"env:{name}",
                "ok": bool(value),
                "value": value,
            }
        )
    return checks


def check_path(
    name: str,
    path: Path,
    *,
    expect_dir: bool = False,
    expect_file: bool = False,
) -> dict[str, Any]:
    exists = path.exists()
    ok = exists
    if expect_dir:
        ok = ok and path.is_dir()
    if expect_file:
        ok = ok and path.is_file()
    return {
        "name": name,
        "ok": ok,
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "is_file": path.is_file() if exists else False,
    }


def check_target_questions(manifest_path: Path, target_ids: set[str]) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return [
            {
                "name": "target_questions",
                "ok": False,
                "missing": sorted(target_ids),
                "message": "source manifest is missing",
            }
        ]
    rows = load_per_question_manifest(manifest_path)
    present = {str(row["question_id"]) for row in rows}
    missing = sorted(target_ids - present)
    return [
        {
            "name": "target_questions",
            "ok": not missing,
            "expected_count": len(target_ids),
            "missing": missing,
        }
    ]


def check_prompts(prompts_dir: str, *, expected_count: int) -> list[dict[str, Any]]:
    try:
        templates = discover_prompt_templates(
            prompts_dir=prompts_dir,
            expected_count=expected_count,
        )
    except Exception as exc:
        return [
            {
                "name": "prompts",
                "ok": False,
                "prompts_dir": prompts_dir,
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]
    return [
        {
            "name": "prompts",
            "ok": True,
            "prompts_dir": prompts_dir,
            "prompt_ids": [str(template["prompt_id"]) for template in templates],
        }
    ]


def check_model_snapshot(model_name: str, cache_dir: str) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        snapshot = Path(snapshot_path)
        required_any = [
            ("config.json",),
            ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"),
            ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"),
        ]
        missing_groups = [
            candidates
            for candidates in required_any
            if not any((snapshot / candidate).exists() for candidate in candidates)
        ]
        return {
            "name": "model_snapshot",
            "ok": not missing_groups,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "snapshot_path": str(snapshot),
            "missing_required_any": missing_groups,
        }
    except Exception as exc:
        return {
            "name": "model_snapshot",
            "ok": False,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _normalize_question_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("gsm8k_platinum_"):
        return text
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        raise ValueError(f"Could not normalize question id: {value!r}")
    return f"gsm8k_platinum_{int(digits):04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/targeted_rerun_short_long_relaxed.yaml",
        help="Targeted rerun config path.",
    )
    parser.add_argument(
        "--prompts-dir",
        default="prompts/GSM8k",
        help="Prompt directory used by targeted generation.",
    )
    parser.add_argument(
        "--source-run-name",
        default="per-question-0419_143551",
        help="Existing source run under ${SCRATCH}/runs.",
    )
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help="Optional explicit source run directory.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=list(DEFAULT_TARGET_QUESTION_CODES),
        help="Target question id/code. Defaults to the 10 selected questions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
