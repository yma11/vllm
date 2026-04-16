#include "layout.hpp"
#include <algorithm>

namespace deep_ep {

namespace layout {

// Host function: launch SYCL kernel
void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         sycl::queue& q) {
    // Configuration parameters
    constexpr int kNumThreads = 256;
    constexpr int kNumExpertsPerEU = 4;
    constexpr int kNumRanksPerEU = 8;

    // Calculate number of workgroups
    int num_expert_blocks = (num_experts + kNumExpertsPerEU - 1) / kNumExpertsPerEU;
    int num_rank_blocks = (num_ranks + kNumRanksPerEU - 1) / kNumRanksPerEU;
    int num_workgroups = num_expert_blocks + num_rank_blocks;

    // Static assertions
    static_assert(kNumRanksPerEU % NUM_MAX_IPC_PEERS == 0, "Invalid number of ranks per EU");
    static_assert(kNumExpertsPerEU <= kNumThreads, "Too many experts per EU");
    static_assert(kNumRanksPerEU <= kNumThreads, "Too many ranks per EU");

    // Configure SYCL nd_range
    sycl::nd_range<1> nd_range(
        sycl::range<1>(num_workgroups * kNumThreads),  // global size
        sycl::range<1>(kNumThreads)                     // local size (workgroup size)
    );

    // Submit kernel
    q.submit([&](sycl::handler& h) {
        // Allocate local memory in handler (not in kernel)
        auto local_mem_expert = sycl::local_accessor<int, 2>(
            sycl::range<2>(kNumThreads, kNumExpertsPerEU), h);
        auto local_mem_rank = sycl::local_accessor<int, 2>(
            sycl::range<2>(kNumThreads, kNumRanksPerEU), h);
        auto local_mem_rdma = sycl::local_accessor<int, 2>(
            sycl::range<2>(kNumThreads, kNumRanksPerEU / NUM_MAX_IPC_PEERS), h);

        h.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto wg_id = static_cast<int>(item.get_group(0));
            auto thread_id = static_cast<int>(item.get_local_id(0));

            // ========== Expert Statistics Section ==========
            int expert_begin_idx = wg_id * kNumExpertsPerEU;
            int expert_end_idx = sycl::min(expert_begin_idx + kNumExpertsPerEU, num_experts);
            
            if (expert_begin_idx < expert_end_idx) {
                // Initialize per-thread counters
                for (int i = 0; i < kNumExpertsPerEU; ++i) {
                    local_mem_expert[thread_id][i] = 0;
                }

                // Each thread scans all tokens
                for (int i = thread_id; i < num_tokens; i += kNumThreads) {
                    auto shifted_topk_idx = topk_idx + i * num_topk;
                    
                    for (int j = 0; j < num_topk; ++j) {
                        int expert_idx = static_cast<int>(shifted_topk_idx[j]);
                        if (expert_begin_idx <= expert_idx && expert_idx < expert_end_idx) {
                            ++local_mem_expert[thread_id][expert_idx - expert_begin_idx];
                        }
                    }
                }

                // Synchronize
                item.barrier(sycl::access::fence_space::local_space);

                // Reduce and sum
                if (expert_begin_idx + thread_id < expert_end_idx) {
                    int sum = 0;
                    for (int i = 0; i < kNumThreads; ++i) {
                        sum += local_mem_expert[i][thread_id];
                    }
                    num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
                }
                return;
            }

            // ========== Rank Statistics Section ==========
            constexpr int kNumRDMARanksPerEU = kNumRanksPerEU / NUM_MAX_IPC_PEERS;
            
            int rank_begin_idx = (wg_id - num_expert_blocks) * kNumRanksPerEU;
            int rank_end_idx = sycl::min(rank_begin_idx + kNumRanksPerEU, num_ranks);
            int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_IPC_PEERS;
            int rdma_rank_end_idx = rank_end_idx / NUM_MAX_IPC_PEERS;

            if (rank_begin_idx < rank_end_idx) {
                const auto num_expert_per_rank = num_experts / num_ranks;
                auto expert_begin = rank_begin_idx * num_expert_per_rank;
                auto expert_end = rank_end_idx * num_expert_per_rank;

                // Initialize per-thread counters
                for (int i = 0; i < kNumRanksPerEU; ++i) {
                    local_mem_rank[thread_id][i] = 0;
                }
                for (int i = 0; i < kNumRDMARanksPerEU; ++i) {
                    local_mem_rdma[thread_id][i] = 0;
                }

                // Each thread scans all tokens
                for (int i = thread_id; i < num_tokens; i += kNumThreads) {
                    auto shifted_topk_idx = topk_idx + i * num_topk;
                    int is_in_rank[kNumRanksPerEU] = {0};
                    int is_in_rdma_rank[kNumRDMARanksPerEU] = {0};

                    for (int j = 0; j < num_topk; ++j) {
                        int expert_idx = static_cast<int>(shifted_topk_idx[j]);
                        if (expert_begin <= expert_idx && expert_idx < expert_end) {
                            int rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                            is_in_rank[rank_idx]++;
                            is_in_rdma_rank[rank_idx / NUM_MAX_IPC_PEERS]++;
                        }
                    }

                    // Update statistics
                    auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
                    for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                        local_mem_rank[thread_id][j] += (is_in_rank[j] > 0);
                    }

                    for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j) {
                        local_mem_rdma[thread_id][j] += (is_in_rdma_rank[j] > 0);
                    }
                }

                // Synchronize
                item.barrier(sycl::access::fence_space::local_space);

                // Reduce and sum (rank)
                if (rank_begin_idx + thread_id < rank_end_idx) {
                    int sum = 0;
                    for (int i = 0; i < kNumThreads; ++i) {
                        sum += local_mem_rank[i][thread_id];
                    }
                    num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
                }

                // Reduce and sum (rdma_rank)
                if (num_tokens_per_rdma_rank != nullptr && rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
                    int sum = 0;
                    for (int i = 0; i < kNumThreads; ++i) {
                        sum += local_mem_rdma[i][thread_id];
                    }
                    num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
                }
            }
        });
    });

    // Wait for kernel completion
    q.wait();
}

}  // namespace layout

}  // namespace deep_ep
