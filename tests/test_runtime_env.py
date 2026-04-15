"""Tests for runtime device selection across heterogeneous accelerators."""

from src.common.runtime_env import (
    build_unsupported_cuda_message,
    get_supported_cuda_arches,
    is_device_capability_supported,
    select_runtime_device,
)


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        arches: list[str] | None = None,
        device_name: str = "Fake GPU",
        capability: tuple[int, int] = (8, 0),
    ) -> None:
        self._available = available
        self._arches = arches or []
        self._device_name = device_name
        self._capability = capability

    def is_available(self) -> bool:
        return self._available

    def get_arch_list(self) -> list[str]:
        return list(self._arches)

    def current_device(self) -> int:
        return 0

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._device_name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return self._capability


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


def test_get_supported_cuda_arches_normalizes_and_sorts() -> None:
    fake_torch = _FakeTorch(
        _FakeCuda(
            available=True,
            arches=["sm_86", "sm_75", "compute_80", "sm_86"],
        )
    )

    assert get_supported_cuda_arches(fake_torch) == ("7.5", "8.0", "8.6")


def test_is_device_capability_supported_accepts_same_family_minor_upgrade() -> None:
    assert is_device_capability_supported((8, 9), ("8.0", "8.6")) is True


def test_is_device_capability_supported_rejects_known_excluded_capability() -> None:
    assert is_device_capability_supported((8, 7), ("8.0", "8.6")) is False


def test_select_runtime_device_uses_cpu_when_forced() -> None:
    fake_torch = _FakeTorch(_FakeCuda(available=True, arches=["sm_80"]))

    selection = select_runtime_device(fake_torch, force_device="cpu")

    assert selection.resolved_device == "cpu"
    assert "forced" in selection.reason.lower()


def test_select_runtime_device_falls_back_to_cpu_when_cuda_missing() -> None:
    fake_torch = _FakeTorch(_FakeCuda(available=False))

    selection = select_runtime_device(fake_torch, force_device="auto")

    assert selection.resolved_device == "cpu"
    assert "not available" in selection.reason.lower()


def test_select_runtime_device_raises_on_unsupported_gpu_without_fallback() -> None:
    fake_torch = _FakeTorch(
        _FakeCuda(
            available=True,
            arches=["sm_75", "sm_80", "sm_86"],
            device_name="Tesla V100-SXM2-32GB",
            capability=(7, 0),
        )
    )

    try:
        select_runtime_device(fake_torch, force_device="auto", allow_cpu_fallback=False)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Unsupported GPUs should raise without CPU fallback.")

    assert "Tesla V100-SXM2-32GB" in message
    assert "7.0" in message
    assert "8.6" in message


def test_select_runtime_device_falls_back_to_cpu_for_unsupported_gpu_when_enabled() -> None:
    fake_torch = _FakeTorch(
        _FakeCuda(
            available=True,
            arches=["sm_75", "sm_80", "sm_86"],
            device_name="Tesla V100-SXM2-32GB",
            capability=(7, 0),
        )
    )

    selection = select_runtime_device(
        fake_torch,
        force_device="auto",
        allow_cpu_fallback=True,
    )

    assert selection.resolved_device == "cpu"
    assert "falling back to cpu" in selection.reason.lower()


def test_build_unsupported_cuda_message_is_actionable() -> None:
    message = build_unsupported_cuda_message(
        gpu_name="Tesla V100-SXM2-32GB",
        gpu_compute_capability="7.0",
        supported_cuda_arches=("7.5", "8.0", "8.6"),
    )

    assert "Request a newer compatible GPU in PBS" in message
    assert "PEAK_COT_ALLOW_CPU_FALLBACK=1" in message
