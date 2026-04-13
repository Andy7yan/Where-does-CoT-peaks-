"""Curation helpers for Stage 1 data-phase artifacts."""

from __future__ import annotations

from collections import Counter
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from src.coarse_analysis import (
    DIFFICULTY_ORDER,
    build_accuracy_buckets_by_difficulty,
    build_coarse_analysis,
    build_question_metadata_v4,
    dedupe_traces_for_analysis,
)
from src.nldd import summarize_corruption_records
from src.reports import build_accuracy_buckets, load_stage1_traces, select_l_star
from src.settings import ExperimentConfig, require_config_value


CANONICAL_ROOT_ITEMS = {
    "accuracy_by_length.csv",
    "coarse_analysis.json",
    "corruptted_traces",
    "lstar_summary.csv",
    "question_metadata.jsonl",
    "run_meta.json",
    "traces.jsonl",
}
GENERATED_ROOT_ITEMS = {
    "README.md",
    "data_phase_manifest.json",
    "legacy",
}
CORRUPTION_MODES = ("all_steps", "sampled_steps")


def curate_data_phase(
    canonical_run_dir: str,
    *,
    legacy_run_dir: str,
    config_path: str = "configs/stage1.yaml",
) -> dict[str, Any]:
    """Curate one canonical data-phase directory for downstream analysis."""

    run_path = Path(canonical_run_dir)
    external_legacy_path = Path(legacy_run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Canonical run directory does not exist: {run_path}")
    if not external_legacy_path.exists():
        raise FileNotFoundError(f"Legacy run directory does not exist: {external_legacy_path}")

    validation = validate_canonical_data_phase(
        run_path,
        config_path=config_path,
    )
    moved_items = move_noncanonical_root_items(run_path)

    manifest = build_data_phase_manifest(
        run_path,
        external_legacy_path=external_legacy_path,
        validation=validation,
        moved_items=moved_items,
    )
    manifest_path = run_path / "data_phase_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_path = run_path / "README.md"
    readme_path.write_text(
        build_overview_markdown(
            external_legacy_path=external_legacy_path,
            validation=validation,
            moved_items=moved_items,
        ),
        encoding="utf-8",
    )

    legacy_dir = run_path / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_readme_path = legacy_dir / "README.md"
    legacy_readme_path.write_text(
        build_legacy_markdown(
            run_path,
            external_legacy_path=external_legacy_path,
            moved_items=moved_items,
        ),
        encoding="utf-8",
    )

    return {
        "canonical_run_dir": str(run_path),
        "legacy_run_dir": str(external_legacy_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "legacy_readme_path": str(legacy_readme_path),
        "moved_items": moved_items,
        "validation": validation,
    }


def validate_canonical_data_phase(
    run_path: Path,
    *,
    config_path: str,
) -> dict[str, Any]:
    """Validate the canonical data-phase artifacts used by analysis."""

    traces = load_stage1_traces(run_path)
    if not traces:
        raise ValueError("Canonical data phase is missing traces.")

    trace_ids = [str(trace["trace_id"]) for trace in traces]
    duplicate_trace_ids = sorted(
        trace_id
        for trace_id, count in Counter(trace_ids).items()
        if count > 1
    )
    if duplicate_trace_ids:
        sample = ", ".join(duplicate_trace_ids[:5])
        raise ValueError(f"Canonical traces contain duplicate trace_id values: {sample}")

    question_ids = sorted({str(trace["question_id"]) for trace in traces})
    metadata_path = run_path / "question_metadata.jsonl"
    metadata_rows = _load_jsonl(metadata_path)
    metadata_question_ids = sorted(str(row["question_id"]) for row in metadata_rows)
    if question_ids != metadata_question_ids:
        missing_in_metadata = sorted(set(question_ids) - set(metadata_question_ids))
        extra_in_metadata = sorted(set(metadata_question_ids) - set(question_ids))
        raise ValueError(
            "question_metadata.jsonl coverage does not match traces.jsonl: "
            f"missing={missing_in_metadata[:5]}, extra={extra_in_metadata[:5]}"
        )

    _validate_accuracy_csv(run_path, traces=traces, config_path=config_path)
    for filename in ("coarse_analysis.json", "lstar_summary.csv"):
        if not (run_path / filename).exists():
            raise FileNotFoundError(f"Canonical data phase is missing required coarse-analysis artifact: {filename}")
    corruption_validation = _validate_corruption_summary(run_path)
    run_meta_notes = _build_run_meta_notes(
        run_path=run_path,
        traces=traces,
        config_path=config_path,
    )

    return {
        "trace_count": len(traces),
        "question_count": len(question_ids),
        "trace_id_unique": True,
        "question_metadata_matches_traces": True,
        "accuracy_csv_matches_traces": True,
        "corruption_summary_matches_records": True,
        "run_meta_notes": run_meta_notes,
        "corruption_validation": corruption_validation,
    }


def move_noncanonical_root_items(run_path: Path) -> list[dict[str, str]]:
    """Move non-canonical root items into run_path/legacy."""

    legacy_dir = run_path / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    allowed = CANONICAL_ROOT_ITEMS | GENERATED_ROOT_ITEMS
    moved_items: list[dict[str, str]] = []
    for path in sorted(run_path.iterdir(), key=lambda item: item.name):
        if path.name in allowed:
            continue
        destination = legacy_dir / path.name
        if destination.exists():
            raise FileExistsError(
                f"Legacy destination already exists for '{path.name}': {destination}"
            )
        _move_path(path, destination)
        moved_items.append(
            {
                "from": path.name,
                "to": str(Path("legacy") / path.name).replace("\\", "/"),
            }
        )
    return moved_items


def build_data_phase_manifest(
    run_path: Path,
    *,
    external_legacy_path: Path,
    validation: dict[str, Any],
    moved_items: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the machine-readable manifest describing data-phase artifacts."""

    artifacts: list[dict[str, Any]] = []
    artifacts.extend(_build_canonical_entries(run_path))
    artifacts.extend(_build_local_legacy_entries(run_path))
    artifacts.extend(_build_external_legacy_entries(run_path, external_legacy_path))

    return {
        "phase": "data_phase",
        "canonical_run_dir": str(run_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_analysis_inputs": [
            "traces.jsonl",
            "question_metadata.jsonl",
            "accuracy_by_length.csv",
            "coarse_analysis.json",
            "lstar_summary.csv",
            "run_meta.json",
            "corruptted_traces/all_steps.jsonl",
            "corruptted_traces/sampled_steps.jsonl",
            "corruptted_traces/corruption_summary.json",
        ],
        "legacy_sources": [
            str(run_path / "legacy"),
            str(external_legacy_path),
        ],
        "moved_items": moved_items,
        "validation": validation,
        "artifacts": artifacts,
    }


def build_overview_markdown(
    *,
    external_legacy_path: Path,
    validation: dict[str, Any],
    moved_items: list[dict[str, str]],
) -> str:
    """Build the top-level analysis-facing overview."""

    notes = "\n".join(f"- {note}" for note in validation["run_meta_notes"])
    moved = "\n".join(f"- `{item['from']}` -> `{item['to']}`" for item in moved_items)
    if not moved:
        moved = "- 本次整理没有发现需要下沉的根目录文件。"
    corruption = validation["corruption_validation"]
    return (
        "# Data Phase Overview\n\n"
        "这个目录现在是 analysis 阶段唯一的正式 data-phase 入口。\n\n"
        "## 默认读取\n\n"
        "- `traces.jsonl`: dedup 后的正式 trace 表\n"
        "- `question_metadata.jsonl`: 基于 dedup trace 生成的题目级元数据\n"
        "- `accuracy_by_length.csv`: 基于 dedup trace 生成的长度-准确率聚合表\n"
        "- `run_meta.json`: 生成时配置快照，不保证与当前 `configs/stage1.yaml` 完全一致\n"
        "- `corruptted_traces/all_steps.jsonl`\n"
        "- `corruptted_traces/sampled_steps.jsonl`\n"
        "- `corruptted_traces/corruption_summary.json`: corruption 正式统计口径\n\n"
        "## 不再默认读取\n\n"
        "- `legacy/failed_corruptions.jsonl` 是旧口径步骤失败记录，不代表当前正式 corruption 成败率\n"
        f"- `{external_legacy_path.as_posix()}` 仅作 provenance / 历史追溯，不作默认 analysis 输入\n\n"
        "## 校验结果\n\n"
        f"- trace 数: `{validation['trace_count']}`\n"
        f"- question 数: `{validation['question_count']}`\n"
        "- `trace_id` 唯一性: 通过\n"
        "- `question_metadata.jsonl` 覆盖一致性: 通过\n"
        "- `accuracy_by_length.csv` 与 dedup trace 一致性: 通过\n"
        "- `corruption_summary.json` 与 corruption records 一致性: 通过\n"
        f"- `all_steps` 记录数 / 失败数: `{corruption['all_steps']['records']}` / `{corruption['all_steps']['failures']}`\n"
        f"- `sampled_steps` 记录数 / 失败数: `{corruption['sampled_steps']['records']}` / `{corruption['sampled_steps']['failures']}`\n\n"
        "## run_meta 说明\n\n"
        f"{notes}\n\n"
        "## 本次下沉到 legacy 的内容\n\n"
        f"{moved}\n"
    )


def build_legacy_markdown(
    run_path: Path,
    *,
    external_legacy_path: Path,
    moved_items: list[dict[str, str]],
) -> str:
    """Build the legacy-area README."""

    moved = "\n".join(f"- `{item['to']}`" for item in moved_items)
    if not moved:
        moved = "- 本次整理没有移动本地文件。"
    return (
        "# Legacy Data-Phase Artifacts\n\n"
        "这个目录保存已经降级为 legacy 的本地产物。它们保留用于人工追溯，但不作为 analysis 默认输入。\n\n"
        "## 本地 legacy\n\n"
        f"{moved}\n\n"
        "## 外部 provenance\n\n"
        f"- `{external_legacy_path.as_posix()}`: 原始 balanced / pre-curation 快照，仅作历史来源\n"
        f"- canonical 目录: `{run_path.as_posix()}`\n"
    )


def _build_canonical_entries(run_path: Path) -> list[dict[str, Any]]:
    return [
        _artifact_entry(
            run_path=run_path,
            path=run_path / "traces.jsonl",
            role="canonical",
            default_for_analysis=True,
            join_keys=["trace_id", "question_id", "prompt_id"],
            source="canonical run root",
            notes="Deduplicated formal trace table used as the default analysis input.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "question_metadata.jsonl",
            role="canonical",
            default_for_analysis=True,
            join_keys=["question_id"],
            source="derived from canonical traces",
            notes="Per-question metadata derived from the deduplicated trace table.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "accuracy_by_length.csv",
            role="canonical",
            default_for_analysis=True,
            join_keys=["bucket_label"],
            source="derived from canonical traces",
            notes="Per-difficulty accuracy-by-length aggregation with raw and dedup views.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "coarse_analysis.json",
            role="canonical",
            default_for_analysis=True,
            join_keys=[],
            source="derived from canonical traces",
            notes="Frozen v4 coarse-analysis boundaries for difficulty, tertiles, and near-L* windows.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "lstar_summary.csv",
            role="canonical",
            default_for_analysis=True,
            join_keys=["difficulty"],
            source="derived from coarse analysis",
            notes="Per-difficulty L* and selected near-L* window summary.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "run_meta.json",
            role="canonical",
            default_for_analysis=True,
            join_keys=[],
            source="generation-time snapshot",
            notes="Generation-time run metadata snapshot; may differ from the current config YAML.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "corruptted_traces" / "all_steps.jsonl",
            role="canonical",
            default_for_analysis=True,
            join_keys=["corruption_id", "trace_id"],
            source="deduplicated corruption records",
            notes="Full corruption records across all eligible steps.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "corruptted_traces" / "sampled_steps.jsonl",
            role="canonical",
            default_for_analysis=True,
            join_keys=["corruption_id", "trace_id"],
            source="deduplicated corruption records",
            notes="Sampled corruption records used for spot checks.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "corruptted_traces" / "corruption_summary.json",
            role="canonical",
            default_for_analysis=True,
            join_keys=[],
            source="derived from canonical corruption records",
            notes="Official corruption success/failure summary for analysis.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "README.md",
            role="derived",
            default_for_analysis=False,
            join_keys=[],
            source="generated during curation",
            notes="Human-readable overview for analysis-stage consumers.",
        ),
        _artifact_entry(
            run_path=run_path,
            path=run_path / "data_phase_manifest.json",
            role="derived",
            default_for_analysis=False,
            join_keys=[],
            source="generated during curation",
            notes="Machine-readable manifest describing canonical and legacy artifacts.",
        ),
    ]


def _build_local_legacy_entries(run_path: Path) -> list[dict[str, Any]]:
    legacy_dir = run_path / "legacy"
    if not legacy_dir.exists():
        return []

    notes_by_name = {
        "failed_corruptions.jsonl": "Legacy step-level failure records retained for provenance only.",
        "manual_review_items.md": "Stale manual review notes from pre-curation reporting.",
        "stage1_analysis_report.md": "Stale analysis report kept only for provenance.",
        "logs": "Operational logs retained for provenance.",
        "shards": "Original shard outputs retained for provenance.",
        "README.md": "Overview of legacy artifacts under this directory.",
    }

    entries: list[dict[str, Any]] = []
    for path in sorted(legacy_dir.iterdir(), key=lambda item: item.name):
        entries.append(
            _artifact_entry(
                run_path=run_path,
                path=path,
                role="legacy",
                default_for_analysis=False,
                join_keys=_default_join_keys_for_path(path),
                source="moved from canonical run root during curation",
                notes=notes_by_name.get(path.name, "Legacy artifact retained for provenance."),
            )
        )
    return entries


def _build_external_legacy_entries(run_path: Path, external_legacy_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in (
        "traces.jsonl",
        "question_metadata.jsonl",
        "accuracy_by_length.csv",
        "coarse_analysis.json",
        "lstar_summary.csv",
        "run_meta.json",
        "failed_corruptions.jsonl",
        "corruptted_traces/all_steps.jsonl",
        "corruptted_traces/sampled_steps.jsonl",
        "corruptted_traces/corruption_summary.json",
    ):
        path = external_legacy_path / relative
        if not path.exists():
            continue
        entries.append(
            _artifact_entry(
                run_path=run_path,
                path=path,
                role="legacy",
                default_for_analysis=False,
                join_keys=_default_join_keys_for_path(path),
                source=f"external legacy snapshot: {external_legacy_path.as_posix()}",
                notes="Historical run snapshot retained for provenance and cross-checking only.",
            )
        )
    return entries


def _artifact_entry(
    *,
    run_path: Path,
    path: Path,
    role: str,
    default_for_analysis: bool,
    join_keys: list[str],
    source: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "path": _manifest_path(run_path, path),
        "role": role,
        "phase": "data_phase",
        "default_for_analysis": default_for_analysis,
        "record_count": _record_count(path),
        "join_keys": join_keys,
        "source": source,
        "notes": notes,
    }


def _default_join_keys_for_path(path: Path) -> list[str]:
    name = path.name
    if name == "traces.jsonl":
        return ["trace_id", "question_id", "prompt_id"]
    if name == "question_metadata.jsonl":
        return ["question_id"]
    if name == "accuracy_by_length.csv":
        return ["difficulty", "dedup_mode", "bucket_label"]
    if name == "lstar_summary.csv":
        return ["difficulty"]
    if name in {"all_steps.jsonl", "sampled_steps.jsonl"}:
        return ["corruption_id", "trace_id"]
    if name == "failed_corruptions.jsonl":
        return ["trace_id", "question_id", "step_index"]
    return []


def _validate_accuracy_csv(run_path: Path, *, traces: list[dict[str, Any]], config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    min_bin_size = require_config_value(
        "analysis.min_bin_size",
        config.analysis.min_bin_size,
    )
    deduped_traces = dedupe_traces_for_analysis(traces)
    question_metadata = build_question_metadata_v4(
        traces=traces,
        deduped_traces=deduped_traces,
        difficulty_quantiles=require_config_value(
            "analysis.difficulty_quantiles",
            config.analysis.difficulty_quantiles,
        ),
        accuracy_exclusion_bounds=require_config_value(
            "analysis.accuracy_exclusion_bounds",
            config.analysis.accuracy_exclusion_bounds,
        ),
    )
    buckets_by_difficulty = build_accuracy_buckets_by_difficulty(
        traces=traces,
        deduped_traces=deduped_traces,
        question_metadata=question_metadata,
        build_accuracy_buckets=build_accuracy_buckets,
        min_bin_size=min_bin_size,
    )
    expected_rows: list[dict[str, str]] = []
    for difficulty in DIFFICULTY_ORDER:
        for dedup_mode in ("dedup", "raw"):
            for bucket in buckets_by_difficulty[difficulty][dedup_mode]:
                expected_rows.append(
                    {
                        "difficulty": difficulty,
                        "dedup_mode": dedup_mode,
                        "bucket_label": _format_bucket_label(bucket.bucket_label),
                        "n": str(bucket.n),
                        "mean": f"{bucket.mean:.6f}",
                        "se": f"{bucket.se:.6f}",
                    }
                )

    accuracy_path = run_path / "accuracy_by_length.csv"
    with accuracy_path.open("r", encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.DictReader(handle))
    if actual_rows != expected_rows:
        raise ValueError("accuracy_by_length.csv is inconsistent with the deduplicated traces.")


def _validate_corruption_summary(run_path: Path) -> dict[str, dict[str, int]]:
    corruption_dir = run_path / "corruptted_traces"
    summary_path = corruption_dir / "corruption_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing corruption summary: {summary_path}")

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = loaded.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("corruption_summary.json is missing a summary mapping.")

    validation: dict[str, dict[str, int]] = {}
    mode_rows: dict[str, list[dict[str, Any]]] = {}
    for mode_name in CORRUPTION_MODES:
        mode_path = corruption_dir / f"{mode_name}.jsonl"
        rows = _load_jsonl(mode_path)
        mode_rows[mode_name] = rows
        failure_count = sum(1 for row in rows if row.get("corruption_failed"))
        expected = summary.get(mode_name)
        if not isinstance(expected, dict):
            raise ValueError(f"corruption_summary.json is missing '{mode_name}'.")
        if expected.get("records") != len(rows) or expected.get("failures") != failure_count:
            raise ValueError(
                f"corruption_summary.json is inconsistent with {mode_name}.jsonl."
            )
        validation[mode_name] = {
            "records": len(rows),
            "failures": failure_count,
        }

    recomputed = json.loads(json.dumps(summarize_corruption_records(mode_rows)))
    if recomputed != summary:
        raise ValueError("corruption_summary.json does not match the recomputed corruption summary.")
    return validation


def _build_run_meta_notes(
    *,
    run_path: Path,
    traces: list[dict[str, Any]],
    config_path: str,
) -> list[str]:
    config = ExperimentConfig.from_yaml(config_path)
    run_meta_path = run_path / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    notes = [
        "当前 `run_meta.json` 被视为生成时配置快照，analysis 应以该文件解释正式 run 的生成参数。",
    ]
    if config.dataset.subset_size is not None:
        actual_questions = len({str(trace["question_id"]) for trace in traces})
        if actual_questions != config.dataset.subset_size:
            notes.append(
                f"当前 config 的 `dataset.subset_size={config.dataset.subset_size}`，"
                f"但 canonical traces 实际覆盖 `{actual_questions}` 题。"
            )

    configured_max_new_tokens = config.generation.max_new_tokens
    run_meta_max_new_tokens = run_meta.get("max_new_tokens")
    if configured_max_new_tokens != run_meta_max_new_tokens:
        notes.append(
            f"当前 config 的 `generation.max_new_tokens={configured_max_new_tokens}`，"
            f"但 run_meta 记录的是 `{run_meta_max_new_tokens}`。"
        )

    configured_sample_counts = config.generation.icl_group_sample_counts
    run_meta_sample_counts = run_meta.get("icl_group_sample_counts") or {}
    if configured_sample_counts != run_meta_sample_counts:
        notes.append(
            "当前 config 的 per-prompt sample counts 与 run_meta 不一致；"
            "analysis 默认只使用 dedup 后 canonical traces，不再追求 balanced 15 traces/question 口径。"
        )
    return notes


def _manifest_path(run_path: Path, path: Path) -> str:
    try:
        return path.relative_to(run_path).as_posix()
    except ValueError:
        return path.relative_to(run_path.parent).as_posix()


def _record_count(path: Path) -> int | None:
    if not path.exists() or path.is_dir():
        return None
    if path.suffix == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            row_count = sum(1 for _ in reader)
        return max(row_count - 1, 0)
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _format_bucket_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and _move_path_windows(source, destination):
        return
    for attempt in range(5):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt == 4:
                break
            time.sleep(0.1)

    try:
        shutil.move(str(source), str(destination))
        return
    except PermissionError:
        pass

    if source.is_dir():
        shutil.copytree(source, destination)
        _remove_path(source)
        return

    shutil.copy2(source, destination)
    _remove_path(source)


def _remove_path(path: Path) -> None:
    for attempt in range(5):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.chmod(path, 0o666)
                path.unlink()
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def _move_path_windows(source: Path, destination: Path) -> bool:
    flags = 0x1 | 0x2  # MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED
    result = ctypes.windll.kernel32.MoveFileExW(str(source), str(destination), flags)
    return bool(result)
