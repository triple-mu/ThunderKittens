#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_to_all_new {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

template <int NUM_DEVICES, bool SCATTER_N_GATHER_S>
struct globals {
    static_assert(NUM_DEVICES == 2 || NUM_DEVICES == 4 || NUM_DEVICES == 8,
        "NUM_DEVICES must be 2, 4, or 8");

    static constexpr int S_BLOCK_SIZE = 16;
    static constexpr int D_BLOCK_SIZE = 64;

    // Tensor view is [B, S, N, D] in memory.
    // We tile over (S, D) using axis=DEPTH, and keep N untiled on rows.
    // That avoids requiring N % 16 == 0.
    using shared_tile = st_bf<S_BLOCK_SIZE, D_BLOCK_SIZE>;
    using parallel_layout = pgl<
        gl<bf16, -1, -1, -1, -1, tma::descriptor<shared_tile, dim::DEPTH>>,
        NUM_DEVICES,
        false
    >;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(
            input.cols() / D_BLOCK_SIZE,
            input.depth() / S_BLOCK_SIZE,
            input.batch() * input.rows()
        );
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

template <int NUM_DEVICES, bool SCATTER_N_GATHER_S>
static inline void check_shapes(const globals<NUM_DEVICES, SCATTER_N_GATHER_S> &G) {
    using G_t = globals<NUM_DEVICES, SCATTER_N_GATHER_S>;

    TORCH_CHECK(G.input.batch() == G.output.batch(), "Batch dimension mismatch");
    TORCH_CHECK(G.input.cols() == G.output.cols(), "D dimension mismatch");

    TORCH_CHECK(
        G.input.cols() == 64 || G.input.cols() == 128 || G.input.cols() == 256,
        "D must be one of {64, 128, 256}"
    );
    TORCH_CHECK(
        G.input.cols() % G_t::D_BLOCK_SIZE == 0,
        "D must be divisible by ", G_t::D_BLOCK_SIZE
    );

    TORCH_CHECK(
        G.input.depth() % G_t::S_BLOCK_SIZE == 0 && G.output.depth() % G_t::S_BLOCK_SIZE == 0,
        "S dimensions must be divisible by ", G_t::S_BLOCK_SIZE
    );

    if constexpr (SCATTER_N_GATHER_S) {
        // [B, S_local, N_global, D] -> [B, S_global, N_local, D]
        TORCH_CHECK(
            G.output.depth() == G.input.depth() * NUM_DEVICES,
            "For (scatter=2, gather=1): output.s must equal input.s * NUM_DEVICES"
        );
        TORCH_CHECK(
            G.input.rows() % NUM_DEVICES == 0,
            "For (scatter=2, gather=1): input.n must be divisible by NUM_DEVICES"
        );
        TORCH_CHECK(
            G.output.rows() == G.input.rows() / NUM_DEVICES,
            "For (scatter=2, gather=1): output.n must equal input.n / NUM_DEVICES"
        );
    } else {
        // [B, S_global, N_local, D] -> [B, S_local, N_global, D]
        TORCH_CHECK(
            G.input.depth() == G.output.depth() * NUM_DEVICES,
            "For (scatter=1, gather=2): input.s must equal output.s * NUM_DEVICES"
        );
        TORCH_CHECK(
            G.output.rows() == G.input.rows() * NUM_DEVICES,
            "For (scatter=1, gather=2): output.n must equal input.n * NUM_DEVICES"
        );
    }
}

template <int NUM_DEVICES, bool SCATTER_N_GATHER_S>
__device__ inline void kernel(const globals<NUM_DEVICES, SCATTER_N_GATHER_S> &G) {
    using G_t = globals<NUM_DEVICES, SCATTER_N_GATHER_S>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G_t::shared_tile &tile =
        allocator.template allocate<typename G_t::shared_tile>();

    const int d_block_idx = static_cast<int>(blockIdx.x);
    const int s_block_idx = static_cast<int>(blockIdx.y);

    const int z_idx = static_cast<int>(blockIdx.z);
    const int n_size = G.input.rows();
    const int batch_idx = z_idx / n_size;
    const int n_idx = z_idx - batch_idx * n_size;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
        tile,
        G.input[G.dev_idx],
        {batch_idx, s_block_idx, n_idx, d_block_idx},
        arrived
    );

    int out_s_block_idx;
    int out_n_idx;
    int dst_dev_idx;

    if constexpr (SCATTER_N_GATHER_S) {
        // scatter axis: N(rows), gather axis: S(depth)
        const int s_local_blocks = G.input.depth() / G_t::S_BLOCK_SIZE;
        const int n_local = G.output.rows();

        dst_dev_idx = n_idx / n_local;
        out_n_idx = n_idx - dst_dev_idx * n_local;
        out_s_block_idx = G.dev_idx * s_local_blocks + s_block_idx;
    } else {
        // scatter axis: S(depth), gather axis: N(rows)
        const int s_local_blocks = G.output.depth() / G_t::S_BLOCK_SIZE;

        dst_dev_idx = s_block_idx / s_local_blocks;
        out_s_block_idx = s_block_idx - dst_dev_idx * s_local_blocks;
        out_n_idx = G.dev_idx * G.input.rows() + n_idx;
    }

    wait(arrived, 0);
    tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
        G.output[dst_dev_idx],
        tile,
        {batch_idx, out_s_block_idx, out_n_idx, d_block_idx}
    );
}

} // namespace all_to_all_new

namespace all_to_all_new_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

template <int NUM_DEVICES>
struct globals {
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

template <int NUM_DEVICES>
__device__ inline void kernel(const globals<NUM_DEVICES> &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace all_to_all_new_barrier

template <int NUM_DEVICES, bool SCATTER_N_GATHER_S>
static inline void launch_mode(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    using all_to_all_G_t = all_to_all_new::globals<NUM_DEVICES, SCATTER_N_GATHER_S>;
    using barrier_G_t = all_to_all_new_barrier::globals<NUM_DEVICES>;

    all_to_all_G_t all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all_G_t::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all_G_t::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };

    all_to_all_new::check_shapes(all_to_all_G);

    barrier_G_t barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<
        all_to_all_new_barrier::config,
        barrier_G_t,
        all_to_all_new_barrier::kernel<NUM_DEVICES>
    >(barrier_G);

    kittens::py::launch_kernel<
        all_to_all_new::config,
        all_to_all_G_t,
        all_to_all_new::kernel<NUM_DEVICES, SCATTER_N_GATHER_S>
    >(all_to_all_G);

    kittens::py::launch_kernel<
        all_to_all_new_barrier::config,
        barrier_G_t,
        all_to_all_new_barrier::kernel<NUM_DEVICES>
    >(barrier_G);
}

template <int NUM_DEVICES>
static inline void dispatch_num_devices(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    if (scatter_axis == 2 && gather_axis == 1) {
        launch_mode<NUM_DEVICES, true>(output, input, barrier);
    } else if (scatter_axis == 1 && gather_axis == 2) {
        launch_mode<NUM_DEVICES, false>(output, input, barrier);
    } else {
        TORCH_CHECK(
            false,
            "all_to_all_new only supports (gather_axis=1, scatter_axis=2) "
            "or (gather_axis=2, scatter_axis=1)"
        );
    }
}

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    TORCH_CHECK(input.data_.dim() == 4, "input must be a 4D tensor");
    TORCH_CHECK(output.data_.dim() == 4, "output must be a 4D tensor");

    TORCH_CHECK(
        (scatter_axis == 2 && gather_axis == 1) ||
        (scatter_axis == 1 && gather_axis == 2),
        "all_to_all_new only supports (gather_axis=1, scatter_axis=2) "
        "or (gather_axis=2, scatter_axis=1)"
    );

    kittens::py::parallel_tensor_check(output, input, barrier);

    const int world_size = input.local_world_size_;
    TORCH_CHECK(
        world_size == output.local_world_size_ && world_size == barrier.local_world_size_,
        "All tensors must have the same local_world_size"
    );

    if (world_size == 2) {
        dispatch_num_devices<2>(output, input, barrier, scatter_axis, gather_axis);
    } else if (world_size == 4) {
        dispatch_num_devices<4>(output, input, barrier, scatter_axis, gather_axis);
    } else if (world_size == 8) {
        dispatch_num_devices<8>(output, input, barrier, scatter_axis, gather_axis);
    } else {
        TORCH_CHECK(false, "all_to_all_new only supports local_world_size in {2, 4, 8}");
    }
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all_new", &entrypoint);
    // Alias for compatibility with existing scripts.
    m.def("tk_all_to_all", &entrypoint);
}
