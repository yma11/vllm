#pragma once

#define NUM_MAX_IPC_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull
#endif

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cassert>

#define EP_HOST_ASSERT(condition) assert(condition)
#define EP_STATIC_ASSERT(condition, message) static_assert(condition, message)

typedef struct { int x, y, z, w; } int4;

template<typename T>
constexpr T ceil_div(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}

namespace deep_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;
#undef INT_BITS_T
#undef INT_BITS_T2

using StreamType = sycl::queue;
using DataType = std::nullptr_t;

struct Config {
    int num_eus;
    int num_max_ipc_chunked_send_tokens;
    int num_max_ipc_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_eus,
           int num_max_ipc_chunked_send_tokens,
           int num_max_ipc_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens=0,
           int num_max_rdma_chunked_recv_tokens=0)
        : num_eus(num_eus),
          num_max_ipc_chunked_send_tokens(num_max_ipc_chunked_send_tokens),
          num_max_ipc_chunked_recv_tokens(num_max_ipc_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_eus >= 0);
        EP_HOST_ASSERT(num_max_ipc_chunked_send_tokens > 0 and num_max_ipc_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_ipc_chunked_send_tokens < num_max_ipc_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens == 0 and num_max_rdma_chunked_recv_tokens == 0);
    }

    size_t get_ipc_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        const int num_channels = num_eus / 2;
        size_t num_bytes = num_channels * num_ranks * num_max_ipc_chunked_recv_tokens * hidden_bytes;
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
    }

};

}  // namespace deep_ep
