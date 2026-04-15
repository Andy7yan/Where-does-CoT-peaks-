"""Tests for generation-side trace assembly and JSONL helpers."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

from scripts.run_generation import (
    build_run_metadata,
    discover_prompt_templates,
    resolve_prompt_sample_counts,
    resolve_prompt_temperatures,
)
from src.generation import (
    TRACE_SCHEMA_VERSION,
    GenerationOutput,
    LLMGenerator,
    _looks_like_tensor,
    _move_model_inputs_to_device,
    _snapshot_has_required_files,
    append_traces_to_jsonl,
    generate_traces_for_question,
    load_existing_trace_ids,
    validate_output_dir_schema,
    write_run_metadata,
)


class FakeGenerator:
    """Small stand-in generator for local unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        self.calls += 1
        return GenerationOutput(
            raw_completion=(
                "Step 1: Break the arithmetic into parts.\n"
                "Step 2: Finish the arithmetic.\n"
                "#### 4"
            ),
            token_count=7,
        )


class BatchFakeGenerator:
    """Batch-capable stand-in generator that records the temperatures it sees."""

    def __init__(self) -> None:
        self.batch_calls: list[tuple[int, float]] = []

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list[GenerationOutput]:
        assert max_new_tokens == 64
        self.batch_calls.append((len(messages_batch), temperature))
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


def test_generate_traces_for_question_builds_trace_schema() -> None:
    generator = FakeGenerator()
    prompt_templates = [
        {
            "prompt_id": "icl_short",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
        {
            "prompt_id": "icl_medium",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
    ]

    traces = generate_traces_for_question(
        generator=generator,
        question_id="gsm8k_0001",
        question_text="What is 2 + 2?",
        gold_answer=4.0,
        prompt_templates=prompt_templates,
        samples_per_group=2,
        temperature=0.7,
        max_new_tokens=64,
    )

    assert len(traces) == 4
    assert traces[0]["trace_id"] == "gsm8k_0001_icl_short_1"
    assert traces[-1]["trace_id"] == "gsm8k_0001_icl_medium_2"
    assert traces[0]["prompt_id"] == "icl_short"
    assert traces[0]["actual_num_steps"] == 2
    assert traces[0]["final_answer_line"] == "#### 4"
    assert traces[0]["extracted_answer"] == 4.0
    assert traces[0]["is_correct"] is True
    assert traces[0]["token_count"] == 7
    assert "timestamp" in traces[0]
    assert set(traces[0]) == {
        "trace_id",
        "question_id",
        "question_text",
        "gold_answer",
        "prompt_id",
        "raw_completion",
        "steps",
        "actual_num_steps",
        "final_answer_line",
        "extracted_answer",
        "is_correct",
        "extraction_failed",
        "token_count",
        "timestamp",
    }
    assert generator.calls == 4


def test_load_existing_trace_ids_returns_empty_for_missing_file() -> None:
    assert load_existing_trace_ids("tests/_missing_trace_file.jsonl") == set()


def test_append_traces_to_jsonl_and_reload_ids() -> None:
    temp_dir = Path("tests") / f"_tmp_generation_{uuid.uuid4().hex}"
    output_path = temp_dir / "traces.jsonl"
    traces = [
        {"trace_id": "trace_1", "value": 1},
        {"trace_id": "trace_2", "value": 2},
    ]

    try:
        append_traces_to_jsonl(traces, str(output_path))
        lines = output_path.read_text(encoding="utf-8").splitlines()
        loaded = [json.loads(line) for line in lines]
        existing_ids = load_existing_trace_ids(str(output_path))

        assert loaded == traces
        assert existing_ids == {"trace_1", "trace_2"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_run_metadata_writes_required_fields() -> None:
    run_metadata = {
        "run_id": "stage-c-run",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset": "madrylab/gsm8k-platinum:test",
        "temperature": None,
        "icl_group_temperatures": {"icl_short": 0.3, "icl_medium": 0.5},
        "max_new_tokens": 128,
        "num_icl_groups": 3,
        "samples_per_group": 4,
        "seed": 42,
        "prompt_ids": ["icl_short", "icl_medium", "icl_detailed"],
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": "2026-04-11T00:00:00Z",
    }

    temp_dir = Path("tests") / f"_tmp_run_metadata_{uuid.uuid4().hex}"
    try:
        meta_path = write_run_metadata(str(temp_dir), run_metadata)
        loaded = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        assert loaded == run_metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_output_dir_schema_rejects_missing_meta() -> None:
    temp_dir = Path("tests") / f"_tmp_schema_missing_meta_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "traces.jsonl").write_text('{"trace_id": "x"}\n', encoding="utf-8")

        try:
            validate_output_dir_schema(str(temp_dir), expected_schema_version=TRACE_SCHEMA_VERSION)
        except RuntimeError as exc:
            assert "missing run_meta.json" in str(exc)
        else:
            raise AssertionError("validate_output_dir_schema should reject traces without metadata")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_output_dir_schema_rejects_incompatible_schema() -> None:
    temp_dir = Path("tests") / f"_tmp_schema_bad_version_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "traces.jsonl").write_text('{"trace_id": "x"}\n', encoding="utf-8")
        (temp_dir / "run_meta.json").write_text(
            json.dumps({"schema_version": "stage1_trace_v1"}),
            encoding="utf-8",
        )

        try:
            validate_output_dir_schema(str(temp_dir), expected_schema_version=TRACE_SCHEMA_VERSION)
        except RuntimeError as exc:
            assert "schema_version" in str(exc)
        else:
            raise AssertionError("validate_output_dir_schema should reject incompatible metadata")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_has_required_files_detects_complete_layout() -> None:
    temp_dir = Path("tests") / f"_tmp_snapshot_complete_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
            (temp_dir / filename).write_text("{}", encoding="utf-8")

        assert _snapshot_has_required_files(temp_dir) is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_has_required_files_detects_incomplete_layout() -> None:
    temp_dir = Path("tests") / f"_tmp_snapshot_incomplete_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "config.json").write_text("{}", encoding="utf-8")

        assert _snapshot_has_required_files(temp_dir) is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_move_model_inputs_to_device_accepts_dict_like_with_to() -> None:
    class FakeBatch(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": "ids", "attention_mask": "mask"})
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    batch = FakeBatch()
    result = _move_model_inputs_to_device(batch, "cuda:0")

    assert batch.seen_device == "cuda:0"
    assert result["input_ids"] == "ids"
    assert result["attention_mask"] == "mask"


def test_move_model_inputs_to_device_accepts_mapping_like_object() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 3)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    class FakeBatchEncoding:
        def __init__(self) -> None:
            self.payload = {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
            }
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

        def items(self):
            return self.payload.items()

    batch = FakeBatchEncoding()
    result = _move_model_inputs_to_device(batch, "cpu")

    assert batch.seen_device == "cpu"
    assert result["input_ids"].seen_device == "cpu"
    assert result["attention_mask"].seen_device == "cpu"


def test_move_model_inputs_to_device_accepts_tensor_like() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 3)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    tensor = FakeTensor()
    result = _move_model_inputs_to_device(tensor, "cpu")

    assert result["input_ids"] is tensor
    assert tensor.seen_device == "cpu"


def test_looks_like_tensor_checks_shape_and_to() -> None:
    class Tensorish:
        shape = (1,)

        def to(self, device):
            return self

    class NotTensorish:
        pass

    assert _looks_like_tensor(Tensorish()) is True
    assert _looks_like_tensor(NotTensorish()) is False


def test_prepare_model_inputs_falls_back_to_string_tokenization() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 4)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=None, return_tensors=None):
            self.calls.append(f"template:{tokenize}")
            if tokenize:
                raise TypeError("unsupported backend")
            return "rendered prompt"

        def __call__(self, prompt, return_tensors="pt", add_special_tokens=False):
            self.calls.append("tokenizer_call")
            assert prompt == "rendered prompt"
            return {"input_ids": FakeTensor()}

    generator = object.__new__(LLMGenerator)
    generator.tokenizer = FakeTokenizer()
    generator.device = "cpu"

    result = LLMGenerator._prepare_model_inputs(generator, [{"role": "user", "content": "hi"}])

    assert generator.tokenizer.calls == ["template:True", "template:False", "tokenizer_call"]
    assert "input_ids" in result
    assert result["input_ids"].seen_device == "cpu"


def test_discover_prompt_templates_sorts_and_validates_count() -> None:
    temp_dir = Path("tests") / f"_tmp_prompt_discovery_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "icl_z.yaml").write_text(
            'prompt_id: "icl_z"\nversion: 1\nsystem: "z"\nfew_shot: []\nuser_template: "{question}"\n',
            encoding="utf-8",
        )
        (temp_dir / "icl_a.yaml").write_text(
            'prompt_id: "icl_a"\nversion: 1\nsystem: "a"\nfew_shot: []\nuser_template: "{question}"\n',
            encoding="utf-8",
        )

        templates = discover_prompt_templates(str(temp_dir), expected_count=2)

        assert [template["prompt_id"] for template in templates] == ["icl_a", "icl_z"]

        try:
            discover_prompt_templates(str(temp_dir), expected_count=3)
        except ValueError as exc:
            assert "Expected 3 ICL prompt groups" in str(exc)
        else:
            raise AssertionError("discover_prompt_templates should validate prompt counts")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_discover_prompt_templates_uses_preferred_prompt_order() -> None:
    temp_dir = Path("tests") / f"_tmp_prompt_preferred_order_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        for prompt_id in ("icl_verbose", "icl_short", "icl_minimal"):
            (temp_dir / f"{prompt_id}.yaml").write_text(
                f'prompt_id: "{prompt_id}"\nversion: 1\nsystem: "s"\nfew_shot: []\nuser_template: "{{question}}"\n',
                encoding="utf-8",
            )

        templates = discover_prompt_templates(
            str(temp_dir),
            expected_count=3,
            preferred_prompt_ids=["icl_minimal", "icl_short", "icl_verbose"],
        )

        assert [template["prompt_id"] for template in templates] == [
            "icl_minimal",
            "icl_short",
            "icl_verbose",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_run_metadata_contains_required_stage_c_fields() -> None:
    config = SimpleNamespace(
        experiment=SimpleNamespace(run_id="stage-c-run", seed=42),
        model=SimpleNamespace(name="meta-llama/Llama-3.1-8B-Instruct"),
        dataset=SimpleNamespace(name="madrylab/gsm8k-platinum", split="test"),
    )

    metadata = build_run_metadata(
        config=config,
        prompt_ids=["icl_short", "icl_medium"],
        temperature=None,
        icl_group_temperatures={"icl_short": 0.3, "icl_medium": 0.5},
        icl_group_sample_counts={"icl_short": 3, "icl_medium": 5},
        max_new_tokens=96,
        num_icl_groups=2,
        samples_per_group=3,
    )

    assert metadata["run_id"] == "stage-c-run"
    assert metadata["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert metadata["dataset"] == "madrylab/gsm8k-platinum:test"
    assert metadata["temperature"] is None
    assert metadata["icl_group_temperatures"] == {
        "icl_short": 0.3,
        "icl_medium": 0.5,
    }
    assert metadata["icl_group_sample_counts"] == {
        "icl_short": 3,
        "icl_medium": 5,
    }
    assert metadata["max_new_tokens"] == 96
    assert metadata["num_icl_groups"] == 2
    assert metadata["samples_per_group"] == 3
    assert metadata["seed"] == 42
    assert metadata["prompt_ids"] == ["icl_short", "icl_medium"]
    assert metadata["schema_version"] == TRACE_SCHEMA_VERSION
    assert "timestamp" in metadata


def test_generate_traces_for_question_batches_samples_with_generate_batch() -> None:
    generator = BatchFakeGenerator()
    prompt_templates = [
        {
            "prompt_id": "icl_short",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        }
    ]

    traces = generate_traces_for_question(
        generator=generator,
        question_id="gsm8k_0001",
        question_text="What is 2 + 2?",
        gold_answer=4.0,
        prompt_templates=prompt_templates,
        samples_per_group=5,
        max_new_tokens=64,
        temperature=0.7,
        batch_size=4,
    )

    assert len(traces) == 5
    assert generator.batch_calls == [(4, 0.7), (1, 0.7)]


def test_generate_traces_for_question_uses_prompt_specific_temperatures() -> None:
    generator = BatchFakeGenerator()
    prompt_templates = [
        {
            "prompt_id": "icl_short",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
        {
            "prompt_id": "icl_medium",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
    ]

    traces = generate_traces_for_question(
        generator=generator,
        question_id="gsm8k_0001",
        question_text="What is 2 + 2?",
        gold_answer=4.0,
        prompt_templates=prompt_templates,
        samples_per_group=2,
        max_new_tokens=64,
        temperature=None,
        prompt_temperatures={"icl_short": 0.3, "icl_medium": 0.5},
        batch_size=4,
    )

    assert len(traces) == 4
    assert generator.batch_calls == [(2, 0.3), (2, 0.5)]


def test_resolve_prompt_temperatures_uses_group_values_and_default_fallback() -> None:
    resolved = resolve_prompt_temperatures(
        prompt_ids=["icl_short", "icl_medium"],
        default_temperature=0.7,
        configured_group_temperatures={"icl_short": 0.3},
    )

    assert resolved == {"icl_short": 0.3, "icl_medium": 0.7}


def test_resolve_prompt_temperatures_requires_complete_coverage_without_default() -> None:
    try:
        resolve_prompt_temperatures(
            prompt_ids=["icl_short", "icl_medium"],
            default_temperature=None,
            configured_group_temperatures={"icl_short": 0.3},
        )
    except ValueError as exc:
        assert "icl_medium" in str(exc)
    else:
        raise AssertionError("resolve_prompt_temperatures should reject uncovered prompts")


def test_resolve_prompt_sample_counts_uses_group_values_and_default_fallback() -> None:
    resolved = resolve_prompt_sample_counts(
        prompt_ids=["icl_short", "icl_medium", "icl_verbose"],
        default_samples_per_group=3,
        configured_group_sample_counts={"icl_verbose": 5},
    )

    assert resolved == {
        "icl_short": 3,
        "icl_medium": 3,
        "icl_verbose": 5,
    }


def test_generate_traces_for_question_uses_prompt_specific_sample_counts() -> None:
    generator = BatchFakeGenerator()
    prompt_templates = [
        {
            "prompt_id": "icl_short",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
        {
            "prompt_id": "icl_verbose",
            "system": "Solve the user's problem clearly.",
            "few_shot": [],
            "user_template": "{question}",
        },
    ]

    traces = generate_traces_for_question(
        generator=generator,
        question_id="gsm8k_0001",
        question_text="What is 2 + 2?",
        gold_answer=4.0,
        prompt_templates=prompt_templates,
        samples_per_group=3,
        prompt_sample_counts={"icl_verbose": 5},
        max_new_tokens=64,
        temperature=0.7,
        batch_size=4,
    )

    assert len(traces) == 8
    assert generator.batch_calls == [(3, 0.7), (4, 0.7), (1, 0.7)]
