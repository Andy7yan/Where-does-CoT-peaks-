"""Non-PBS PrOntoQA environment and code-consistency preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_generation import discover_prompt_templates
from src.common.reasoning import extract_choice_answer
from src.common.settings import ExperimentConfig
from src.data_phase1.tasks import get_prompts_dir, get_task_name, load_question_records_for_config


KEY_PATHS = [
    "configs/stage1_prontoqa.yaml",
    "jobs/generate_prontoqa.pbs",
    "jobs/submit_generate_prontoqa.sh",
    "jobs/prontoqa_data_phase.pbs",
    "jobs/profile_prontoqa_difficulty.pbs",
    "prompts/PrOntoQA/pqa_short.yaml",
    "prompts/PrOntoQA/pqa_medium.yaml",
    "prompts/PrOntoQA/pqa_detailed.yaml",
    "prompts/PrOntoQA/pqa_verbose.yaml",
    "scripts/run_generation.py",
    "scripts/check_generation_preflight.py",
    "scripts/print_dataset_size.py",
    "scripts/check_prontoqa_generation_quality.py",
    "scripts/check_prontoqa_env_consistency.py",
    "src/common/reasoning.py",
    "src/common/settings.py",
    "src/common/prontoqa_paper_corruption.py",
    "src/data_phase1/tasks.py",
    "src/data_phase1/prontoqa_paper.py",
    "src/data_phase1/generation.py",
    "src/data_phase1/prompting.py",
    "src/data_phase2/coarse_analysis.py",
    "src/data_phase2/difficulty_groups.py",
    "src/data_phase2/pipeline.py",
    "src/data_phase2/postprocess.py",
    "src/data_phase2/difficulty_profile.py",
]


def main() -> None:
    args = parse_args()
    report = build_report(config_path=args.config, check_torch=args.check_torch)
    if args.expected_signature:
        report["signature_comparison"] = compare_signature(
            current=report["file_signature"],
            expected_path=Path(args.expected_signature),
        )
    if args.write_signature:
        output_path = Path(args.write_signature)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report["file_signature"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["wrote_signature"] = str(output_path)

    report["ok"] = report_is_ok(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def build_report(*, config_path: str, check_torch: bool) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(config_path)
    task_name = get_task_name(config)
    prompts_dir = get_prompts_dir(config)
    prompt_templates = discover_prompt_templates(
        prompts_dir=prompts_dir,
        expected_count=config.generation.num_icl_groups or 0,
        preferred_prompt_ids=config.generation.icl_group_prompt_ids or None,
    )
    questions = load_question_records_for_config(config=config)
    extraction = extract_choice_answer("Reasoning...\nFinal Answer: True")

    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
            "env": {
                key: os.environ.get(key)
                for key in ("SCRATCH", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
            },
        },
        "config": {
            "path": config_path,
            "run_id": config.experiment.run_id,
            "task_name": task_name,
            "dataset": f"{config.dataset.name}:{config.dataset.split}",
            "synthetic_question_count": config.dataset.synthetic_question_count,
            "pronto_min_hops": config.dataset.pronto_min_hops,
            "pronto_max_hops": config.dataset.pronto_max_hops,
            "num_icl_groups": config.generation.num_icl_groups,
            "samples_per_group": config.generation.samples_per_group,
            "temperature": config.generation.temperature,
            "max_new_tokens": config.generation.max_new_tokens,
            "answer_mode": config.answer_extraction.mode,
        },
        "prompts": {
            "dir": prompts_dir,
            "prompt_ids": [template["prompt_id"] for template in prompt_templates],
        },
        "dataset_check": {
            "question_count": len(questions),
            "first_question_id": questions[0]["question_id"] if questions else None,
            "first_gold_answer": questions[0]["gold_answer"] if questions else None,
        },
        "answer_extraction_check": {
            "input": "Final Answer: True",
            "extracted": extraction.value,
            "extraction_failed": extraction.extraction_failed,
        },
        "file_signature": build_file_signature(),
        "torch_check": None,
        "issues": [],
    }
    report["issues"].extend(validate_report(report))
    if check_torch:
        report["torch_check"] = build_torch_check()
    return report


def validate_report(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    env = report["environment"]["env"]
    for key in ("SCRATCH", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        if not env.get(key):
            issues.append(f"missing_env:{key}")
    if report["config"]["task_name"] != "prontoqa_paper":
        issues.append("config_task_is_not_prontoqa_paper")
    if report["config"]["answer_mode"] != "choice_ab":
        issues.append("answer_mode_is_not_choice_ab")
    if report["dataset_check"]["question_count"] != report["config"]["synthetic_question_count"]:
        issues.append("dataset_count_mismatch")
    if report["answer_extraction_check"]["extracted"] != "A":
        issues.append("choice_extraction_true_did_not_map_to_A")
    missing = [
        path
        for path, payload in report["file_signature"]["files"].items()
        if not payload["exists"]
    ]
    for path in missing:
        issues.append(f"missing_file:{path}")
    return issues


def build_file_signature() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative_path in KEY_PATHS:
        path = PROJECT_ROOT / relative_path
        files[relative_path] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "size": path.stat().st_size if path.exists() and path.is_file() else None,
        }
    return {
        "schema_version": "prontoqa_env_signature_v1",
        "files": files,
    }


def compare_signature(*, current: dict[str, Any], expected_path: Path) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    current_files = current.get("files", {})
    expected_files = expected.get("files", {})
    mismatches: list[dict[str, Any]] = []
    for path in sorted(set(current_files) | set(expected_files)):
        current_payload = current_files.get(path)
        expected_payload = expected_files.get(path)
        if current_payload != expected_payload:
            mismatches.append(
                {
                    "path": path,
                    "current": current_payload,
                    "expected": expected_payload,
                }
            )
    return {
        "expected_signature_path": str(expected_path),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def build_torch_check() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "import_ok": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def report_is_ok(report: dict[str, Any]) -> bool:
    if report.get("issues"):
        return False
    comparison = report.get("signature_comparison")
    if comparison and comparison.get("mismatch_count", 0) > 0:
        return False
    torch_check = report.get("torch_check")
    if torch_check and not torch_check.get("import_ok"):
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1_prontoqa.yaml")
    parser.add_argument(
        "--write-signature",
        default=None,
        help="Write key-file SHA256 signature for later comparison on Katana.",
    )
    parser.add_argument(
        "--expected-signature",
        default=None,
        help="Compare current key-file signature against a previously written signature JSON.",
    )
    parser.add_argument(
        "--check-torch",
        action="store_true",
        help="Also import torch and report CUDA devices. This does not load the model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
