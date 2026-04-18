"""Print runtime device compatibility and exit non-zero on unsupported accelerators."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.runtime_env import select_runtime_device


def main() -> int:
    import torch

    selection = select_runtime_device(torch)
    print("runtime_device_requested=", selection.requested_device)
    print("runtime_device_resolved=", selection.resolved_device)
    print("runtime_device_reason=", selection.reason)
    print("runtime_gpu_name=", selection.gpu_name)
    print("runtime_gpu_compute_capability=", selection.gpu_compute_capability)
    print("runtime_torch_supported_cuda_arches=", list(selection.supported_cuda_arches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
