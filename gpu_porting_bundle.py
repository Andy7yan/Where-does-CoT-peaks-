"""
Portable GPU runtime bundle extracted from SAE-v1.

Primary source areas in this repo:
- src/dist_utils.py
- src/train.py
- src/activation_store.py
- src/resumable_version/train_r.py
- src/eval_sae.py
- src/run-1gpu.pbs

Quick source map:
- DDP bootstrap, CUDA device binding, bf16/fp16 choice:
  src/dist_utils.py:25, 44, 47, 50-53, 80
- Pinned CPU staging, non_blocking host->device copies, CUDA timing,
  peak-memory logging, DDP wrapping:
  src/train.py:47-77, 128-146, 201-214, 315-320, 393-416
- Frozen model loading onto GPU and batched dict transfer:
  src/activation_store.py:35-42, 106
- Resume path: load checkpoint on CPU, then move optimizer state to GPU:
  src/resumable_version/train_r.py:156-176
- Eval-time cleanup and single-GPU dtype selection:
  src/eval_sae.py:200, 253-254
- Runtime tuning knobs:
  src/config.py:21, 52-56
- Scheduler launch example:
  src/run-1gpu.pbs:3, 37

What this file keeps:
- GPU / CUDA runtime setup
- torchrun / DDP bootstrap helpers
- bf16 / fp16 dtype selection
- pinned-CPU staging buffer for faster host->device copies
- non_blocking tensor transfer helpers
- CUDA timing and peak-memory helpers
- optimizer-state migration back onto GPU after CPU checkpoint load
- minimal launch templates you can adapt in another project

This file is intentionally self-contained so you can copy it into another
project, such as peak-CoT, without bringing the rest of SAE-v1 along.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


T = TypeVar("T")
TModule = TypeVar("TModule", bound=nn.Module)


TORCHRUN_EXAMPLE = "torchrun --nproc_per_node=4 train.py"

PBS_GPU_JOB_TEMPLATE = """#!/bin/bash
#PBS -N your-job-name
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=11:50:00
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
source .env

torchrun --nproc_per_node=1 train.py
"""


@dataclass(frozen=True)
class RuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    model_dtype: torch.dtype
    distributed: bool


def choose_model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def setup_gpu_runtime(
    *,
    require_cuda: bool = True,
    seed: int | None = None,
    tokenizers_parallelism: str | None = None,
    rayon_num_threads: int | None = None,
    torch_num_threads: int | None = None,
    torch_num_interop_threads: int | None = None,
    init_distributed: bool = True,
    dist_backend: str = "nccl",
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
) -> RuntimeContext:
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError("CUDA is required.")

    if tokenizers_parallelism is not None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", tokenizers_parallelism)
    if rayon_num_threads is not None:
        os.environ.setdefault("RAYON_NUM_THREADS", str(rayon_num_threads))

    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    if torch_num_interop_threads is not None and torch_num_interop_threads > 0:
        torch.set_num_interop_threads(torch_num_interop_threads)

    env_has_torchrun = all(
        key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    distributed = init_distributed and env_has_torchrun

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if has_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=dist_backend)

    if seed is not None:
        torch.manual_seed(seed)
        if has_cuda:
            torch.cuda.manual_seed_all(seed)

    if has_cuda and enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if has_cuda:
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark

    return RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        model_dtype=choose_model_dtype(device),
        distributed=distributed,
    )


def log0(message: str) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}][gpu_bundle] {message}", flush=True)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def module_of(model: TModule | DDP) -> TModule:
    if isinstance(model, DDP):
        return cast(TModule, model.module)
    return model


def maybe_wrap_ddp(model: TModule, runtime: RuntimeContext) -> TModule | DDP:
    if not runtime.distributed:
        return model
    return DDP(
        model,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return x.detach().clone()
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return x.detach().clone()
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y


def all_reduce_min_int(value: int, device: torch.device) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return int(value)
    tensor = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def move_to_device(
    value: T,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> T:
    if isinstance(value, torch.Tensor):
        return cast(T, value.to(device=device, non_blocking=non_blocking))
    if isinstance(value, dict):
        return cast(
            T,
            {
                key: move_to_device(item, device, non_blocking=non_blocking)
                for key, item in value.items()
            },
        )
    if isinstance(value, list):
        return cast(
            T,
            [move_to_device(item, device, non_blocking=non_blocking) for item in value],
        )
    if isinstance(value, tuple):
        return cast(
            T,
            tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value),
        )
    return value


def load_checkpoint_cpu(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


class PinnedTensorBuffer:
    """
    CPU staging buffer mirroring SAE-v1's activation buffering pattern.

    Usage pattern:
    1. Add tensors from GPU work via add().
    2. Keep them on CPU float32, optionally pinning memory.
    3. Transfer minibatches back to GPU with non_blocking=True.
    """

    def __init__(self, *, pin_memory: bool = True, cpu_dtype: torch.dtype = torch.float32):
        self.pin_memory = pin_memory
        self.cpu_dtype = cpu_dtype
        self.chunks: list[torch.Tensor] = []
        self.size = 0

    def add(self, tensor: torch.Tensor, *, scale: float | None = None) -> None:
        if tensor.numel() == 0:
            return
        value = tensor.detach()
        if scale is not None:
            value = value / scale
        value = value.to(device="cpu", dtype=self.cpu_dtype)
        if self.pin_memory:
            value = value.pin_memory()
        self.chunks.append(value)
        self.size += int(value.shape[0])

    def add_masked(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> None:
        valid = x[mask]
        self.add(valid, scale=scale)

    def ready(
        self,
        min_items: int,
        *,
        device: torch.device | None = None,
        sync_across_ranks: bool = False,
    ) -> bool:
        if sync_across_ranks:
            if device is None:
                raise ValueError("device is required when sync_across_ranks=True")
            return all_reduce_min_int(self.size, device) >= min_items
        return self.size >= min_items

    def pop_cpu_batches(
        self,
        *,
        total_items: int,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[torch.Tensor]:
        if total_items > self.size:
            raise ValueError(
                f"Requested {total_items} items from buffer, but only {self.size} are available."
            )

        all_x = torch.cat(self.chunks, dim=0)
        if shuffle:
            perm = torch.randperm(all_x.shape[0])
            all_x = all_x[perm]

        take = all_x[:total_items]
        left = all_x[total_items:]

        self.chunks = [left] if left.shape[0] > 0 else []
        self.size = int(left.shape[0])

        for start in range(0, total_items, batch_size):
            yield take[start : start + batch_size]

    def pop_gpu_batches(
        self,
        *,
        total_items: int,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        non_blocking: bool = True,
    ) -> Iterator[torch.Tensor]:
        for batch in self.pop_cpu_batches(
            total_items=total_items,
            batch_size=batch_size,
            shuffle=shuffle,
        ):
            yield batch.to(device=device, non_blocking=non_blocking)


class CUDATimer:
    def __init__(self, device: torch.device):
        self.device = device

    def measure(self, fn):
        if self.device.type != "cuda":
            start = time.perf_counter()
            out = fn()
            end = time.perf_counter()
            return out, end - start

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = fn()
        end_event.record()
        end_event.synchronize()
        return out, start_event.elapsed_time(end_event) / 1000.0


@torch.no_grad()
def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


@torch.no_grad()
def reset_gpu_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def maybe_empty_cuda_cache(device: torch.device | None = None) -> None:
    if device is None or device.type == "cuda":
        torch.cuda.empty_cache()


def runtime_summary(runtime: RuntimeContext) -> str:
    return (
        f"rank={runtime.rank} "
        f"local_rank={runtime.local_rank} "
        f"world_size={runtime.world_size} "
        f"device={runtime.device} "
        f"model_dtype={runtime.model_dtype} "
        f"distributed={runtime.distributed}"
    )


def example_single_gpu_setup(seed: int = 42) -> RuntimeContext:
    runtime = setup_gpu_runtime(require_cuda=True, seed=seed, init_distributed=False)
    log0(f"Single GPU runtime ready | {runtime_summary(runtime)}")
    return runtime


def example_torchrun_setup(
    seed: int = 42,
    *,
    tokenizers_parallelism: str = "true",
    rayon_num_threads: int = 4,
    torch_num_threads: int = 4,
    torch_num_interop_threads: int = 1,
) -> RuntimeContext:
    runtime = setup_gpu_runtime(
        require_cuda=True,
        seed=seed,
        tokenizers_parallelism=tokenizers_parallelism,
        rayon_num_threads=rayon_num_threads,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
        init_distributed=True,
    )
    log0(f"Torchrun runtime ready | {runtime_summary(runtime)}")
    return runtime


def example_training_loop_bits(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cpu_batches: Sequence[torch.Tensor],
    runtime: RuntimeContext,
) -> None:
    """
    Tiny reference snippet showing the exact transfer pattern SAE-v1 used.
    """

    timer = CUDATimer(runtime.device)
    wrapped_model = maybe_wrap_ddp(model.to(runtime.device), runtime)
    reset_gpu_peak_memory_stats(runtime.device)

    for x_cpu in cpu_batches:
        x_gpu, h2d_time_s = timer.measure(
            lambda batch=x_cpu: batch.to(runtime.device, non_blocking=True)
        )

        def step_fn():
            optimizer.zero_grad(set_to_none=True)
            y = wrapped_model(x_gpu)
            loss = y.mean() if isinstance(y, torch.Tensor) else y[0].mean()
            loss.backward()
            optimizer.step()
            return loss

        loss, step_time_s = timer.measure(step_fn)
        mem = get_gpu_memory_stats(runtime.device)
        log0(
            "train_step "
            f"loss={float(loss.detach().item()):.6f} "
            f"h2d_ms={1000.0 * h2d_time_s:.2f} "
            f"step_ms={1000.0 * step_time_s:.2f} "
            f"peak_alloc_gb={mem['peak_alloc_gb']:.2f}"
        )
