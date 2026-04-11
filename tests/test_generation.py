"""Tests for generation-side trace assembly and JSONL helpers."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

from scripts.run_generation import build_run_metadata, discover_prompt_templates
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
        "dataset": "gsm8k:test",
        "temperature": 0.3,
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


def test_build_run_metadata_contains_required_stage_c_fields() -> None:
    config = SimpleNamespace(
        experiment=SimpleNamespace(run_id="stage-c-run", seed=42),
        model=SimpleNamespace(name="meta-llama/Llama-3.1-8B-Instruct"),
        dataset=SimpleNamespace(name="gsm8k", split="test"),
    )

    metadata = build_run_metadata(
        config=config,
        prompt_ids=["icl_short", "icl_medium"],
        temperature=0.2,
        max_new_tokens=96,
        num_icl_groups=2,
        samples_per_group=3,
    )

    assert metadata["run_id"] == "stage-c-run"
    assert metadata["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert metadata["dataset"] == "gsm8k:test"
    assert metadata["temperature"] == 0.2
    assert metadata["max_new_tokens"] == 96
    assert metadata["num_icl_groups"] == 2
    assert metadata["samples_per_group"] == 3
    assert metadata["seed"] == 42
    assert metadata["prompt_ids"] == ["icl_short", "icl_medium"]
    assert metadata["schema_version"] == TRACE_SCHEMA_VERSION
    assert "timestamp" in metadata
