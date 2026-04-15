"""Runtime device selection helpers for heterogeneous GPU environments."""

from dataclasses import dataclass
import os
from typing import Any


_EXCLUDED_FAMILY_COMPAT_CAPABILITIES = {
    (8, 7),
    (10, 1),
}


@dataclass(frozen=True)
class RuntimeDeviceSelection:
    """Resolved execution device plus enough context for useful logging."""

    requested_device: str
    resolved_device: str
    reason: str
    gpu_name: str | None
    gpu_compute_capability: str | None
    supported_cuda_arches: tuple[str, ...]


def select_runtime_device(
    torch_module: Any,
    *,
    force_device: str | None = None,
    allow_cpu_fallback: bool | None = None,
) -> RuntimeDeviceSelection:
    """Resolve whether generation should run on CUDA or CPU."""

    requested_device = _normalize_requested_device(
        force_device
        if force_device is not None
        else os.getenv("PEAK_COT_FORCE_DEVICE", "auto")
    )
    cpu_fallback_enabled = (
        allow_cpu_fallback
        if allow_cpu_fallback is not None
        else _read_bool_env("PEAK_COT_ALLOW_CPU_FALLBACK")
    )
    supported_arches = get_supported_cuda_arches(torch_module)

    if requested_device == "cpu":
        return RuntimeDeviceSelection(
            requested_device=requested_device,
            resolved_device="cpu",
            reason="CPU execution was forced via PEAK_COT_FORCE_DEVICE=cpu.",
            gpu_name=None,
            gpu_compute_capability=None,
            supported_cuda_arches=supported_arches,
        )

    cuda_available = bool(torch_module.cuda.is_available())
    if not cuda_available:
        if requested_device == "cuda":
            raise RuntimeError(
                "CUDA execution was forced via PEAK_COT_FORCE_DEVICE=cuda, "
                "but torch.cuda.is_available() is False."
            )
        return RuntimeDeviceSelection(
            requested_device=requested_device,
            resolved_device="cpu",
            reason="CUDA is not available in the current runtime, so execution will use CPU.",
            gpu_name=None,
            gpu_compute_capability=None,
            supported_cuda_arches=supported_arches,
        )

    device_index = int(torch_module.cuda.current_device())
    gpu_name = str(torch_module.cuda.get_device_name(device_index))
    capability = tuple(torch_module.cuda.get_device_capability(device_index))
    capability_text = _format_capability(capability)

    if is_device_capability_supported(capability, supported_arches):
        return RuntimeDeviceSelection(
            requested_device=requested_device,
            resolved_device="cuda",
            reason=(
                f"Using CUDA on {gpu_name} (compute capability {capability_text}), "
                "which is compatible with the current PyTorch build."
            ),
            gpu_name=gpu_name,
            gpu_compute_capability=capability_text,
            supported_cuda_arches=supported_arches,
        )

    message = build_unsupported_cuda_message(
        gpu_name=gpu_name,
        gpu_compute_capability=capability_text,
        supported_cuda_arches=supported_arches,
    )
    if requested_device == "cuda" or not cpu_fallback_enabled:
        raise RuntimeError(message)

    return RuntimeDeviceSelection(
        requested_device=requested_device,
        resolved_device="cpu",
        reason=(
            f"{message} Falling back to CPU because "
            "PEAK_COT_ALLOW_CPU_FALLBACK=1 is set."
        ),
        gpu_name=gpu_name,
        gpu_compute_capability=capability_text,
        supported_cuda_arches=supported_arches,
    )


def get_supported_cuda_arches(torch_module: Any) -> tuple[str, ...]:
    """Return normalized CUDA architecture tags compiled into this PyTorch build."""

    raw_arches = getattr(torch_module.cuda, "get_arch_list", lambda: [])() or []
    parsed = []
    seen: set[tuple[int, int]] = set()
    for raw_arch in raw_arches:
        capability = _parse_arch_tag(raw_arch)
        if capability is None or capability in seen:
            continue
        seen.add(capability)
        parsed.append(capability)
    parsed.sort()
    return tuple(_format_capability(capability) for capability in parsed)


def is_device_capability_supported(
    device_capability: tuple[int, int],
    supported_cuda_arches: tuple[str, ...],
) -> bool:
    """Return whether the current PyTorch build should be able to execute on a GPU."""

    if not supported_cuda_arches:
        return True

    normalized_arches = []
    for raw_arch in supported_cuda_arches:
        capability = _parse_arch_tag(raw_arch)
        if capability is not None:
            normalized_arches.append(capability)
    if not normalized_arches:
        return True

    if device_capability in normalized_arches:
        return True
    if device_capability in _EXCLUDED_FAMILY_COMPAT_CAPABILITIES:
        return False

    device_major, device_minor = device_capability
    return any(
        arch_major == device_major and arch_minor <= device_minor
        for arch_major, arch_minor in normalized_arches
    )


def build_unsupported_cuda_message(
    *,
    gpu_name: str,
    gpu_compute_capability: str,
    supported_cuda_arches: tuple[str, ...],
) -> str:
    """Construct a single clear failure message for unsupported CUDA devices."""

    supported = ", ".join(supported_cuda_arches) if supported_cuda_arches else "unknown"
    return (
        f"Current PyTorch build cannot execute CUDA kernels on GPU '{gpu_name}' "
        f"(compute capability {gpu_compute_capability}). "
        f"Supported CUDA architectures in this environment: {supported}. "
        "Request a newer compatible GPU in PBS, install a PyTorch build that supports this GPU, "
        "or set PEAK_COT_ALLOW_CPU_FALLBACK=1 (or PEAK_COT_FORCE_DEVICE=cpu) to run on CPU."
    )


def _normalize_requested_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "PEAK_COT_FORCE_DEVICE must be one of: auto, cpu, cuda."
        )
    return normalized


def _read_bool_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_arch_tag(raw_arch: str) -> tuple[int, int] | None:
    normalized = raw_arch.strip().lower()
    if "." in normalized:
        major_text, _, minor_text = normalized.partition(".")
        if major_text.isdigit() and minor_text.isdigit():
            return int(major_text), int(minor_text)
        return None
    if "_" in normalized:
        normalized = normalized.split("_", 1)[1]
    if len(normalized) < 2 or not normalized.isdigit():
        return None
    if len(normalized) == 2:
        return int(normalized[0]), int(normalized[1])
    return int(normalized[:-1]), int(normalized[-1])


def _format_capability(capability: tuple[int, int]) -> str:
    return f"{capability[0]}.{capability[1]}"


__all__ = [
    "RuntimeDeviceSelection",
    "build_unsupported_cuda_message",
    "get_supported_cuda_arches",
    "is_device_capability_supported",
    "select_runtime_device",
]
