"""Tests for the per-question generation scaffold."""

import json
from pathlib import Path
import shutil
import uuid

from scripts.run_generation import discover_prompt_templates
from scripts.run_generation_per_question import run_per_question_generation
from src.data_phase1.generation import GenerationOutput
from src.data_phase1.gsm8k import build_ranked_questions
from src.data_phase1.per_question_selection import (
    build_per_question_manifest,
    build_per_question_selection_metadata,
    infer_per_question_shard_count,
    plan_per_question_shards,
    resolve_source_run_dir,
    save_per_question_manifest,
)


class FakePerQuestionGenerator:
    """Batch-capable fake generator for per-question tests."""

    def __init__(self, model_name: str, dtype: str = "float16", cache_dir: str | None = None):
        self.model_name = model_name
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.batch_calls: list[int] = []

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list[GenerationOutput]:
        assert temperature == 0.6
        assert max_new_tokens == 1024
        self.batch_calls.append(len(messages_batch))
        return [
            GenerationOutput(
                raw_completion=(
                    "Step 1: Break the arithmetic into parts.\n"
                    "Step 2: Finish the arithmetic.\n"
                    "#### 4"
                ),
                token_count=7,
            )
            for _ in messages_batch
        ]


def test_build_per_question_manifest_selects_medium_and_hard_in_ranked_order(
    monkeypatch,
    sample_gsm8k: list[dict],
) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    temp_dir = Path("tests") / f"_tmp_per_question_manifest_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_data_path = temp_dir / "gsm8k_local.json"
        local_data_path.write_text(
            json.dumps(sample_gsm8k[:6], ensure_ascii=False),
            encoding="utf-8",
        )

        ranked_questions = build_ranked_questions(sample_gsm8k[:6], hash_seed=42)
        metadata_rows = [
            {
                "question_id": ranked_questions[4]["question_id"],
                "difficulty_bucket": "hard",
            },
            {
                "question_id": ranked_questions[1]["question_id"],
                "difficulty_bucket": "easy",
            },
            {
                "question_id": ranked_questions[2]["question_id"],
                "difficulty_bucket": "medium",
            },
        ]
        source_run_dir = temp_dir / "source_run"
        source_run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = source_run_dir / "question_metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in metadata_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest = build_per_question_manifest(
            config_path="configs/stage1_per_question.yaml",
            source_run=str(source_run_dir),
            source="local",
            local_path=str(local_data_path),
        )
        selection_meta = build_per_question_selection_metadata(
            config_path="configs/stage1_per_question.yaml",
            source_run=str(source_run_dir),
            manifest=manifest,
        )

        assert [row["question_id"] for row in manifest] == [
            ranked_questions[2]["question_id"],
            ranked_questions[4]["question_id"],
        ]
        assert [row["source_difficulty_bucket"] for row in manifest] == ["medium", "hard"]
        assert [row["target_total_traces"] for row in manifest] == [120, 300]
        assert [row["target_samples_per_prompt"] for row in manifest] == [30, 75]
        assert selection_meta["selected_question_count"] == 2
        assert selection_meta["difficulty_counts"] == {"medium": 1, "hard": 1}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_per_question_generation_writes_shard_outputs_and_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setattr(
        "scripts.run_generation_per_question.LLMGenerator",
        FakePerQuestionGenerator,
    )

    temp_dir = Path("tests") / f"_tmp_run_per_question_generation_{uuid.uuid4().hex}"
    try:
        output_dir = temp_dir / "run"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "question_id": "gsm8k_platinum_0000",
                "question_text": "What is 2 + 2?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0001",
                "question_text": "What is 3 + 1?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0002",
                "question_text": "What is 1 + 3?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "hard",
                "target_total_traces": 300,
                "target_samples_per_prompt": 75,
            },
        ]
        selection_metadata = {
            "schema_version": "per_question_selection_v1",
            "pipeline_variant": "per_question",
            "source_run_dir": "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235",
            "source_run_name": "generate-rerun-0416_165235",
            "source_question_metadata_path": (
                "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235/question_metadata.jsonl"
            ),
            "selected_question_count": 3,
            "per_question_trace_policy": {
                "medium": {
                    "target_total_traces": 120,
                    "target_samples_per_prompt": 30,
                },
                "hard": {
                    "target_total_traces": 300,
                    "target_samples_per_prompt": 75,
                },
            },
        }
        manifest_path, _ = save_per_question_manifest(
            output_dir,
            manifest=manifest,
            selection_metadata=selection_metadata,
        )

        artifacts = run_per_question_generation(
            config_path="configs/stage1_per_question.yaml",
            output_dir=str(output_dir),
            question_manifest_path=manifest_path,
            prompts_dir="prompts/per_question",
            start_idx=1,
            end_idx=2,
            shard_id="q0001_0002",
        )

        shard_dir = output_dir / "shards" / "q0001_0002"
        traces_path = shard_dir / "traces.jsonl"
        run_meta_path = shard_dir / "run_meta.json"

        assert traces_path.exists()
        assert run_meta_path.exists()
        assert artifacts["pipeline_variant"] == "per_question"
        assert artifacts["written_traces"] == 120
        assert artifacts["total_traces"] == 120

        trace_lines = traces_path.read_text(encoding="utf-8").splitlines()
        assert len(trace_lines) == 120
        trace_rows = [json.loads(line) for line in trace_lines]
        assert {row["question_id"] for row in trace_rows} == {"gsm8k_platinum_0001"}

        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        assert run_meta["pipeline_variant"] == "per_question"
        assert run_meta["prompt_dir"] == "prompts/per_question"
        assert run_meta["source_run_dir"] == (
            "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235"
        )
        assert run_meta["source_run_name"] == "generate-rerun-0416_165235"
        assert run_meta["source_question_metadata_path"] == (
            "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235/question_metadata.jsonl"
        )
        assert run_meta["selected_question_count"] == 3
        assert run_meta["prompt_ids"] == [
            "icl_short",
            "icl_medium",
            "icl_detailed",
            "icl_verbose",
        ]
        assert (output_dir / "per_question_manifest.jsonl").exists()
        assert (output_dir / "per_question_selection_meta.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_per_question_generation_preserves_root_selection_inputs_for_repair(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")
    monkeypatch.setattr(
        "scripts.run_generation_per_question.LLMGenerator",
        FakePerQuestionGenerator,
    )

    temp_dir = Path("tests") / f"_tmp_run_per_question_repair_{uuid.uuid4().hex}"
    try:
        output_dir = temp_dir / "run"
        output_dir.mkdir(parents=True, exist_ok=True)
        full_manifest = [
            {
                "question_id": "gsm8k_platinum_0000",
                "question_text": "What is 2 + 2?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "medium",
                "target_total_traces": 120,
                "target_samples_per_prompt": 30,
            },
            {
                "question_id": "gsm8k_platinum_0001",
                "question_text": "What is 3 + 1?",
                "gold_answer": 4.0,
                "source_difficulty_bucket": "hard",
                "target_total_traces": 300,
                "target_samples_per_prompt": 75,
            },
        ]
        selection_metadata = {
            "schema_version": "per_question_selection_v1",
            "pipeline_variant": "per_question",
            "source_run_dir": "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235",
            "source_run_name": "generate-rerun-0416_165235",
            "source_question_metadata_path": (
                "/srv/scratch/test/peak-CoT/runs/generate-rerun-0416_165235/question_metadata.jsonl"
            ),
            "selected_question_count": 2,
            "per_question_trace_policy": {
                "medium": {
                    "target_total_traces": 120,
                    "target_samples_per_prompt": 30,
                },
                "hard": {
                    "target_total_traces": 300,
                    "target_samples_per_prompt": 75,
                },
            },
        }
        save_per_question_manifest(
            output_dir,
            manifest=full_manifest,
            selection_metadata=selection_metadata,
        )

        repair_dir = temp_dir / "repair"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_manifest = [full_manifest[1]]
        repair_manifest_path, _ = save_per_question_manifest(
            repair_dir,
            manifest=repair_manifest,
            selection_metadata=selection_metadata,
        )

        run_per_question_generation(
            config_path="configs/stage1_per_question.yaml",
            output_dir=str(output_dir),
            question_manifest_path=repair_manifest_path,
            prompts_dir="prompts/per_question",
            start_idx=0,
            end_idx=1,
            shard_id="repair_0000_0001",
            preserve_root_selection_inputs=True,
        )

        persisted_rows = [
            json.loads(line)
            for line in (output_dir / "per_question_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["question_id"] for row in persisted_rows] == [
            "gsm8k_platinum_0000",
            "gsm8k_platinum_0001",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_discover_prompt_templates_supports_legacy_root_fallback() -> None:
    templates = discover_prompt_templates(
        "prompts",
        expected_count=4,
        preferred_prompt_ids=["icl_short", "icl_medium", "icl_detailed", "icl_verbose"],
    )

    assert [template["prompt_id"] for template in templates] == [
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ]


def test_discover_prompt_templates_supports_per_question_prompt_directory() -> None:
    templates = discover_prompt_templates(
        "prompts/per_question",
        expected_count=4,
        preferred_prompt_ids=["icl_short", "icl_medium", "icl_detailed", "icl_verbose"],
    )

    assert [template["prompt_id"] for template in templates] == [
        "icl_short",
        "icl_medium",
        "icl_detailed",
        "icl_verbose",
    ]


def test_resolve_source_run_dir_prefers_scratch_runs_for_results_style_input(
    monkeypatch,
) -> None:
    temp_dir = Path("tests") / f"_tmp_source_run_resolution_{uuid.uuid4().hex}"
    try:
        scratch_root = temp_dir / "scratch"
        source_run_dir = scratch_root / "runs" / "generate-rerun-0416_165235"
        source_run_dir.mkdir(parents=True, exist_ok=True)
        (source_run_dir / "question_metadata.jsonl").write_text(
            '{"question_id":"gsm8k_platinum_0000","difficulty_bucket":"medium"}\n',
            encoding="utf-8",
        )

        monkeypatch.setenv("SCRATCH", str(scratch_root))

        resolved = resolve_source_run_dir("results/generate-rerun-0416_165235")

        assert resolved == source_run_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_infer_per_question_shard_count_uses_target_trace_budget() -> None:
    manifest = (
        [{"target_total_traces": 120} for _ in range(171)]
        + [{"target_total_traces": 300} for _ in range(123)]
    )

    shard_count = infer_per_question_shard_count(
        manifest,
        target_traces_per_shard=7200,
    )

    assert shard_count == 8


def test_plan_per_question_shards_avoids_tiny_remainder_shard() -> None:
    manifest = [{"target_total_traces": 120} for _ in range(10)]

    shard_plan = plan_per_question_shards(manifest, shard_count=3)

    assert [(row["start_idx"], row["end_idx"]) for row in shard_plan] == [
        (0, 3),
        (3, 7),
        (7, 10),
    ]
    assert [row["question_count"] for row in shard_plan] == [3, 4, 3]
    assert [row["target_total_traces"] for row in shard_plan] == [360, 480, 360]
