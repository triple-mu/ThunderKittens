import os
import sys
from functools import lru_cache

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from common import (
    clean_print,
    destroy_distributed_environment,
    init_distributed_environment,
)

def nccl_all_to_all_func(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
) -> torch.Tensor:
    local_world_size = torch.distributed.get_world_size()

    if scatter_idx == 2 and gather_idx == 1:
        # [B, S_local, N_global, D] -> [B, S_global, N_local, D]
        b, s_local, n_global, d = input.shape
        if n_global % local_world_size != 0:
            raise RuntimeError("N_global must be divisible by local_world_size")
        n_local = n_global // local_world_size

        output = torch.empty((b, s_local * local_world_size, n_local, d), dtype=input.dtype, device=input.device)

        input_t = (
            input.view(b, s_local, local_world_size, n_local, d)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        output_t = torch.empty_like(input_t)
        torch.distributed.all_to_all_single(output_t, input_t)

        output.copy_(
            output_t.permute(1, 0, 2, 3, 4)
            .reshape(b, s_local * local_world_size, n_local, d)
        )

    elif scatter_idx == 1 and gather_idx == 2:
        # [B, S_global, N_local, D] -> [B, S_local, N_global, D]
        b, s_global, n_local, d = input.shape
        if s_global % local_world_size != 0:
            raise RuntimeError("S_global must be divisible by local_world_size")
        s_local = s_global // local_world_size

        output = torch.empty((b, s_local, n_local * local_world_size, d), dtype=input.dtype, device=input.device)

        input_t = (
            input.view(b, local_world_size, s_local, n_local, d)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        output_t = torch.empty_like(input_t)
        torch.distributed.all_to_all_single(output_t, input_t)

        output.copy_(
            output_t.permute(1, 2, 0, 3, 4)
            .reshape(b, s_local, n_local * local_world_size, d)
        )

    else:
        raise RuntimeError("Only (scatter=2,gather=1) and (scatter=1,gather=2) are supported")

    return output


@lru_cache(maxsize=1)
def _get_tk_extension():
    try:
        from _C import TKParallelTensor, tk_all_to_all_new as tk_kernel
    except ImportError:
        from _C import TKParallelTensor, tk_all_to_all as tk_kernel
    return TKParallelTensor, tk_kernel


_BARRIER_CACHE = {}


def tk_all_to_all_func(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    if not input.is_cuda:
        raise RuntimeError("input must be a CUDA tensor")
    if input.dim() != 4:
        raise RuntimeError("input must be a 4D tensor")
    if input.dtype != torch.bfloat16:
        raise RuntimeError("input dtype must be torch.bfloat16")

    local_rank = input.device.index
    local_world_size = torch.distributed.get_world_size()

    if local_world_size not in (2, 4, 8):
        raise RuntimeError("tk_all_to_all_func requires world_size in {2, 4, 8}")

    b, s, n, d = input.shape
    if scatter_idx == 2 and gather_idx == 1:
        if n % local_world_size != 0:
            raise RuntimeError("For scatter=2,gather=1, N must be divisible by world_size")
        output_shape = (b, s * local_world_size, n // local_world_size, d)
    elif scatter_idx == 1 and gather_idx == 2:
        if s % local_world_size != 0:
            raise RuntimeError("For scatter=1,gather=2, S must be divisible by world_size")
        output_shape = (b, s // local_world_size, n * local_world_size, d)
    else:
        raise RuntimeError("Only (scatter=2,gather=1) and (scatter=1,gather=2) are supported")

    TKParallelTensor, tk_kernel = _get_tk_extension()

    input_c = input.contiguous()

    # IMPORTANT:
    # Use TK-managed allocations (VMM path) instead of wrapping pre-allocated torch tensors
    # (LEGACY IPC path). For this all-to-all kernel, remote writes are reliable on the VMM path.
    input_tk = TKParallelTensor(
        tuple(input_c.shape),
        dtype=input_c.dtype,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False,
    )
    input_tk.data_.copy_(input_c)

    output_tk = TKParallelTensor(
        output_shape,
        dtype=input.dtype,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False,
    )

    key = (local_rank, local_world_size)
    barrier_tk = _BARRIER_CACHE.get(key)
    if barrier_tk is None:
        barrier_tk = TKParallelTensor(
            (1, 1),
            dtype=torch.int,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        _BARRIER_CACHE[key] = barrier_tk
    barrier_tk.data_.zero_()

    # Wait for all ranks to observe barrier initialization.
    torch.distributed.barrier()
    tk_kernel(output_tk, input_tk, barrier_tk, scatter_idx, gather_idx)
    # Launch is async. First wait local stream completion, then wait all ranks so every
    # peer write into this rank's output is globally complete before wrappers destruct.
    torch.cuda.synchronize(input.device)
    torch.distributed.barrier()

    # Return plain torch.Tensor while keeping API identical to PyTorch-style functions.
    return output_tk.data_.clone()


def run_case(
    local_world_size: int,
    scatter_idx: int,
    gather_idx: int,
    b: int = 1,
    s_local: int = 16,
    n_local: int = 13,
    d: int = 64,
) -> None:
    local_rank = torch.distributed.get_rank()
    device = f"cuda:{local_rank}"

    n_global = n_local * local_world_size
    s_global = s_local * local_world_size

    if scatter_idx == 2 and gather_idx == 1:
        input_shape = (b, s_local, n_global, d)
        output_shape = (b, s_global, n_local, d)
    elif scatter_idx == 1 and gather_idx == 2:
        input_shape = (b, s_global, n_local, d)
        output_shape = (b, s_local, n_global, d)
    else:
        raise RuntimeError("Unsupported scatter/gather pair")

    input_ref = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    output_nccl = nccl_all_to_all_func(input_ref, scatter_idx, gather_idx)
    output_tk = tk_all_to_all_func(input_ref, scatter_idx, gather_idx)
    torch.cuda.synchronize()

    if not torch.equal(output_tk, output_nccl):
        diff = (output_tk - output_nccl).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"Mismatch for scatter={scatter_idx}, gather={gather_idx}. "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    clean_print(
        f"PASS scatter={scatter_idx}, gather={gather_idx}, "
        f"input={input_shape}, output={output_shape}, n_local={n_local}",
        print_once=True,
    )


def main() -> None:
    _, local_world_size = init_distributed_environment()

    if local_world_size not in (2, 4, 8):
        raise RuntimeError("benchmark_new.py requires LOCAL_WORLD_SIZE in {2, 4, 8}")

    try:
        # Keep this intentionally small. We only verify correctness.
        # Cover both aligned and non-aligned S cases.
        run_case(local_world_size, scatter_idx=2, gather_idx=1, s_local=16)
        run_case(local_world_size, scatter_idx=1, gather_idx=2, s_local=16)
        run_case(local_world_size, scatter_idx=2, gather_idx=1, s_local=13)
        run_case(local_world_size, scatter_idx=1, gather_idx=2, s_local=13)
        clean_print("All correctness tests passed.", print_once=True)
    finally:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
