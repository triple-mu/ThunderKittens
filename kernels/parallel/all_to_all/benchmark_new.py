import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from common import (
    clean_print,
    destroy_distributed_environment,
    init_distributed_environment,
)

try:
    from _C import TKParallelTensor, tk_all_to_all_new as _tk_all_to_all
except ImportError:
    from _C import TKParallelTensor, tk_all_to_all as _tk_all_to_all


def nccl_all_to_all_func(
    output: torch.Tensor,
    input: torch.Tensor,
    local_world_size: int,
    scatter_idx: int,
    gather_idx: int,
) -> None:
    if scatter_idx == 2 and gather_idx == 1:
        # [B, S_local, N_global, D] -> [B, S_global, N_local, D]
        b, s_local, n_global, d = input.shape
        if n_global % local_world_size != 0:
            raise RuntimeError("N_global must be divisible by local_world_size")
        n_local = n_global // local_world_size

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


def tk_all_to_all_func(
    output: TKParallelTensor,
    input: TKParallelTensor,
    barrier: TKParallelTensor,
    scatter_idx: int,
    gather_idx: int,
) -> None:
    _tk_all_to_all(output, input, barrier, scatter_idx, gather_idx)


def run_case(
    local_rank: int,
    local_world_size: int,
    scatter_idx: int,
    gather_idx: int,
    b: int = 1,
    s_local: int = 16,
    n_local: int = 13,
    d: int = 64,
) -> None:
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
    output_nccl = torch.empty(output_shape, dtype=torch.bfloat16, device=device)

    input_tk = TKParallelTensor(
        input_shape,
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False,
    )
    output_tk = TKParallelTensor(
        output_shape,
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False,
    )
    barrier_tk = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True,
    )

    input_tk.data_.copy_(input_ref)
    output_tk.data_.zero_()
    barrier_tk.data_.zero_()

    # Wait for barrier memory to be visible on all ranks.
    torch.distributed.barrier()

    nccl_all_to_all_func(output_nccl, input_ref, local_world_size, scatter_idx, gather_idx)
    tk_all_to_all_func(output_tk, input_tk, barrier_tk, scatter_idx, gather_idx)
    torch.cuda.synchronize()

    if not torch.equal(output_tk.data_, output_nccl):
        diff = (output_tk.data_ - output_nccl).abs()
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
    local_rank, local_world_size = init_distributed_environment()

    if local_world_size not in (2, 4, 8):
        raise RuntimeError("benchmark_new.py requires LOCAL_WORLD_SIZE in {2, 4, 8}")

    try:
        # Keep this intentionally small. We only verify correctness.
        run_case(local_rank, local_world_size, scatter_idx=2, gather_idx=1)
        run_case(local_rank, local_world_size, scatter_idx=1, gather_idx=2)
        clean_print("All correctness tests passed.", print_once=True)
    finally:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
