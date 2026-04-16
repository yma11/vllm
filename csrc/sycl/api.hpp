#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include "configs.h"

namespace deep_ep {

namespace internode {

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);
void* alloc(size_t size, size_t alignment);
void free(void* ptr);

}  // namespace internode

namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, sycl::queue& stream);

void barrier_perf_test(int** barrier_signal_ptrs,
                       int rank, int num_ranks,
                       int inner_repeat,
                       sycl::queue& stream);

void barrier_stress_test(void** buffer_ptrs,
                         int** barrier_signal_ptrs,
                         int* error_count,
                         int rank, int num_ranks,
                         int inner_repeat, int iter_offset,
                         int data_size, int data_offset_ints,
                         sycl::queue& stream);

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     sycl::queue& stream,
                     int num_channels);

}  // namespace intranode

}  // namespace deep_ep
