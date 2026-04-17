#pragma once

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <tuple>
#include <vector>

#include <level_zero/ze_api.h>
#include "sycl/configs.h"
#include "event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace shared_memory {

union MemHandleInner {
    ze_ipc_mem_handle_t ze_ipc_mem_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t size;
};

constexpr size_t HANDLE_SIZE = sizeof(MemHandle);

class SharedMemoryAllocator {
public:
    SharedMemoryAllocator() = default;
    SharedMemoryAllocator(sycl::queue& queue);
    void init_from_queue(sycl::queue& queue);
    void malloc(void** ptr, size_t size);
    void free(void* ptr);
    void get_mem_handle(MemHandle* mem_handle, void* ptr);
    void open_mem_handle(void** ptr, MemHandle* mem_handle);
    void close_mem_handle(void* ptr);

private:
    ze_context_handle_t ze_context = nullptr;
    ze_device_handle_t ze_device = nullptr;
};

}  // namespace shared_memory

namespace deep_ep {

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_IPC_PEERS == 8, "The number of maximum IPC peers must be 8");

private:
    bool low_latency_mode = false;

    // IPC buffer
    int64_t num_ipc_bytes;
    void* buffer_ptrs[NUM_MAX_IPC_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // RDMA buffer (reserved for future internode support)
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode
    bool enable_shrink = false;

    // Device info
    int device_id;
    int num_device_eus;
    int rank, rdma_rank, ipc_rank;
    int num_ranks, num_rdma_ranks, num_ipc_ranks;
    shared_memory::MemHandle ipc_handles[NUM_MAX_IPC_PEERS];

    // Communication queue
    sycl::queue comm_stream;

    bool available = false;
    bool explicitly_destroy;
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_IPC_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE counters (shared memory for host-device access)
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    shared_memory::SharedMemoryAllocator shared_memory_allocator;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_ipc_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink);

    ~Buffer() noexcept(false);

    bool is_available() const;
    bool is_internode_available() const;
    int get_num_rdma_ranks() const;
    int get_rdma_rank() const;
    int get_root_rdma_rank(bool global) const;
    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;
    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

    sycl::queue get_comm_queue() const;

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt);

    void destroy();

    // IPC handle exchange via Ring AllGather (Unix socket + SCM_RIGHTS)
    std::vector<std::optional<pybind11::bytearray>> all_gather_handle(
        const pybind11::bytearray& local_ipc_handle,
        const pybind11::function& barrier_func);

    // Layout computation
    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor& topk_idx,
                        int num_experts,
                        std::optional<EventHandle>& previous_event,
                        bool async,
                        bool allocate_on_comm_stream);

    // Barrier test methods
    void test_barrier(const std::optional<c10::intrusive_ptr<c10d::ProcessGroup>>& process_group = std::nullopt);
    void test_barrier_perf(int inner_repeat);
    int test_barrier_stress(int inner_repeat, int iter_offset, int data_size);

    // Intranode dispatch
    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    // Notify dispatch test
    std::tuple<int, std::vector<int>, torch::Tensor, torch::Tensor> test_notify_dispatch(
        const torch::Tensor& num_tokens_per_rank,
        const torch::Tensor& num_tokens_per_expert,
        const torch::Tensor& is_token_in_rank,
        int num_tokens,
        int num_experts,
        int num_channels,
        int expert_alignment);
};

}  // namespace deep_ep
