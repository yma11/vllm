#include <sycl/sycl.hpp>
#include "configs.h"
#include "utils.hpp"

namespace deep_ep {
namespace intranode {

inline void atomic_add_system(int* addr, int val) {
    auto ref = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::system,
                                sycl::access::address_space::global_space>(*addr);
    ref.fetch_add(val);
}

template <int kNumRanks>
class BarrierKernel {
public:
    BarrierKernel(int** barrier_signal_ptrs, int rank)
        : barrier_signal_ptrs_(barrier_signal_ptrs),
          rank_(rank) {}

    void operator()(sycl::nd_item<1> item) const {
        // Use block-bypass barrier
        barrier_block_bypass<kNumRanks, true>(barrier_signal_ptrs_, rank_, item);
    }

private:
    int** barrier_signal_ptrs_;
    int rank_;
};


void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, sycl::queue& stream) {
    constexpr int kNumThreads = 128;
    
    sycl::range<1> global_range(kNumThreads);
    sycl::range<1> local_range(kNumThreads);


    #define BARRIER_LAUNCH_CASE(ranks)                                              \
        case ranks: {                                                               \
            try {                                                                   \
                stream.submit([&](sycl::handler& cgh) {                                \
                                                                                                        cgh.parallel_for(                                              \
                        sycl::nd_range<1>(global_range, local_range),             \
                        [=](sycl::nd_item<1> item) {                              \
                            BarrierKernel<ranks> kernel(                          \
                                barrier_signal_ptrs,                               \
                                rank);                                             \
                            kernel(item);                                          \
                        });                                                        \
                });                                                                \
            } catch (sycl::exception const& e) {                                   \
                throw;                                                             \
            }                                                                      \
            break;                                                                 \
        }

    switch (num_ranks) {
        BARRIER_LAUNCH_CASE(1);
        BARRIER_LAUNCH_CASE(2);
        BARRIER_LAUNCH_CASE(4);
        BARRIER_LAUNCH_CASE(8);
        default:
            EP_HOST_ASSERT(false && "Unsupported number of ranks");
    }

    #undef BARRIER_LAUNCH_CASE
    
    stream.wait();
}

void barrier_perf_test(int** barrier_signal_ptrs,
                       int rank, int num_ranks,
                       int inner_repeat,
                       sycl::queue& stream) {
    for (int i = 0; i < inner_repeat; ++i) {
        barrier(barrier_signal_ptrs, rank, num_ranks, stream);
    }
}


template <int kNumRanks>
class BarrierStressTestKernel {
public:
    BarrierStressTestKernel(
        void** buffer_ptrs,
        int** barrier_signal_ptrs,
        int* error_count,
        int rank,
        int inner_repeat,
        int iter_offset,
        int data_size,
        int data_offset_ints)
        : buffer_ptrs_(buffer_ptrs),
          barrier_signal_ptrs_(barrier_signal_ptrs),
          error_count_(error_count),
          rank_(rank),
          inner_repeat_(inner_repeat),
          iter_offset_(iter_offset),
          data_size_(data_size),
          data_offset_ints_(data_offset_ints) {}

    void operator()(sycl::nd_item<1> item) const {
        auto thread_id = static_cast<int>(item.get_local_id(0));
        auto num_threads = static_cast<int>(item.get_local_range(0));

        for (int i = 0; i < inner_repeat_; ++i) {
            int iter = iter_offset_ + i;

            // Step 1: WRITE — each rank writes to its own IPC data buffer
            int* my_data = reinterpret_cast<int*>(
                static_cast<uint8_t*>(buffer_ptrs_[rank_])) + data_offset_ints_;
            for (int j = thread_id; j < data_size_; j += num_threads) {
                int encoded = iter * 10000 + rank_ * 100 + (j % 100);
                st_volatile_global(my_data + j, encoded);
            }

            // Step 2: Fence data writes, then BARRIER
            memory_fence_system();
            item.barrier(sycl::access::fence_space::local_space);
            barrier_block_bypass<kNumRanks, true>(barrier_signal_ptrs_, rank_, item);

            // Step 3: READ + VERIFY — read all peers' data
            int local_errors = 0;
            for (int p = 0; p < kNumRanks; ++p) {
                if (p == rank_) continue;
                int* peer_data = reinterpret_cast<int*>(
                    static_cast<uint8_t*>(buffer_ptrs_[p])) + data_offset_ints_;
                for (int j = thread_id; j < data_size_; j += num_threads) {
                    int expected = iter * 10000 + p * 100 + (j % 100);
                    int actual = ld_volatile_global(peer_data + j);
                    if (actual != expected) {
                        local_errors++;
                    }
                }
            }
            if (local_errors > 0) {
                atomic_add_system(error_count_, local_errors);
            }

            // Step 4: BARRIER — ensure all reads done before next write
            barrier_block_bypass<kNumRanks, true>(barrier_signal_ptrs_, rank_, item);
        }
    }

private:
    void** buffer_ptrs_;
    int** barrier_signal_ptrs_;
    int* error_count_;
    int rank_;
    int inner_repeat_;
    int iter_offset_;
    int data_size_;
    int data_offset_ints_;
};

void barrier_stress_test(void** buffer_ptrs,
                         int** barrier_signal_ptrs,
                         int* error_count,
                         int rank, int num_ranks,
                         int inner_repeat, int iter_offset,
                         int data_size, int data_offset_ints,
                         sycl::queue& stream) {
    constexpr int kNumThreads = 128;

    sycl::range<1> global_range(kNumThreads);
    sycl::range<1> local_range(kNumThreads);

    #define STRESS_LAUNCH_CASE(ranks)                                               \
        case ranks: {                                                               \
            stream.submit([&](sycl::handler& cgh) {                                \
                cgh.parallel_for(                                                  \
                    sycl::nd_range<1>(global_range, local_range),                  \
                    [=](sycl::nd_item<1> item) {                                   \
                        BarrierStressTestKernel<ranks> kernel(                     \
                            buffer_ptrs, barrier_signal_ptrs,                      \
                            error_count, rank,                                     \
                            inner_repeat, iter_offset,                             \
                            data_size, data_offset_ints);                          \
                        kernel(item);                                              \
                    });                                                            \
            });                                                                    \
            break;                                                                 \
        }

    switch (num_ranks) {
        STRESS_LAUNCH_CASE(2);
        STRESS_LAUNCH_CASE(4);
        STRESS_LAUNCH_CASE(8);
        default:
            EP_HOST_ASSERT(false && "Unsupported number of ranks for stress test");
    }

    #undef STRESS_LAUNCH_CASE

    stream.wait();
}

template <int kNumRanks>
class NotifyDispatchKernel {
public:
    NotifyDispatchKernel(
        const int* num_tokens_per_rank,
        int* moe_recv_counter_mapped,
        const int* num_tokens_per_expert,
        int* moe_recv_expert_counter_mapped,
        int num_experts,
        int num_tokens,
        int num_channels,
        const bool* is_token_in_rank,
        int* channel_prefix_matrix,
        int* rank_prefix_matrix_copy,
        int num_memset_int,
        int expert_alignment,
        void** buffer_ptrs,
        int** barrier_signal_ptrs,
        int rank)
        : num_tokens_per_rank_(num_tokens_per_rank),
          moe_recv_counter_mapped_(moe_recv_counter_mapped),
          num_tokens_per_expert_(num_tokens_per_expert),
          moe_recv_expert_counter_mapped_(moe_recv_expert_counter_mapped),
          num_experts_(num_experts),
          num_tokens_(num_tokens),
          num_channels_(num_channels),
          is_token_in_rank_(is_token_in_rank),
          channel_prefix_matrix_(channel_prefix_matrix),
          rank_prefix_matrix_copy_(rank_prefix_matrix_copy),
          num_memset_int_(num_memset_int),
          expert_alignment_(expert_alignment),
          buffer_ptrs_(buffer_ptrs),
          barrier_signal_ptrs_(barrier_signal_ptrs),
          rank_(rank) {}

    void operator()(sycl::nd_item<1> item) const {
        auto sm_id = static_cast<int>(item.get_group(0));
        auto thread_id = static_cast<int>(item.get_local_id(0));
        auto num_threads = static_cast<int>(item.get_local_range(0));
        auto lane_id = thread_id % 32;
        auto warp_id = thread_id / 32;
        auto num_warps = num_threads / 32;

        if (sm_id == 0) {

            barrier_block_bypass<kNumRanks, true>(barrier_signal_ptrs_, rank_, item);

            int *per_rank_buffer, *per_expert_buffer;
            // Each thread handles one rank
            if (thread_id < kNumRanks) {
                // Write into the rank-to-rank count matrix
                per_rank_buffer = static_cast<int*>(buffer_ptrs_[thread_id]);
                // Per-rank-to-local-expert token counts
                per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
            }

            // After this loop:
            //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
            //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
            int num_experts_per_rank = num_experts_ / kNumRanks;
            if (thread_id < kNumRanks) {
                per_rank_buffer[rank_ * kNumRanks + thread_id] = num_tokens_per_rank_[thread_id];
                #pragma unroll
                for (int i = 0; i < num_experts_per_rank; ++i)
                    per_expert_buffer[rank_ * num_experts_per_rank + i] = 
                        num_tokens_per_expert_[thread_id * num_experts_per_rank + i];
            }

            // Wait for all ranks to finish writing counts
            barrier_block_bypass<kNumRanks>(barrier_signal_ptrs_, rank_, item);

            auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs_[rank_]);
            if (thread_id < kNumRanks) {
                #pragma unroll
                for (int i = 1; i < kNumRanks; ++i)
                    local_per_rank_buffer[i * kNumRanks + thread_id] += 
                        local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
                
                if (thread_id == rank_)
                    *moe_recv_counter_mapped_ = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank_];
            }

            auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
            if (thread_id < num_experts_per_rank) {
                int sum = 0;
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i)
                    sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
                sum = (sum + expert_alignment_ - 1) / expert_alignment_ * expert_alignment_;
                moe_recv_expert_counter_mapped_[thread_id] = sum;
            }

            // Intra-block synchronization
            item.barrier(sycl::access::fence_space::local_space);

            // Copy rank prefix matrix to output tensor
            #pragma unroll
            for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
                rank_prefix_matrix_copy_[i] = local_per_rank_buffer[i];

            // Zero out the buffer for subsequent communication
            #pragma unroll
            for (int i = thread_id; i < num_memset_int_; i += num_threads)
                local_per_expert_buffer[i] = 0;

            item.barrier(sycl::access::fence_space::local_space);

            // Final barrier synchronization
            barrier_block_bypass<kNumRanks>(barrier_signal_ptrs_, rank_, item);
            
            
        } 
        else {
            // ===== Blocks 1..kNumRanks: channel prefix sum computation =====
            
            int dst_rank = sm_id - 1;
            
            // Each warp handles one channel
            for (int channel_id = warp_id; channel_id < num_channels_; channel_id += num_warps) {
                // Compute the token range for this channel
                int token_start_idx, token_end_idx;
                get_channel_task_range(num_tokens_, num_channels_, channel_id, 
                                      token_start_idx, token_end_idx);

                // Count tokens in this channel destined for dst_rank
                // is_token_in_rank shape: [num_tokens, num_ranks]
                int count = 0;
                for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                    count += is_token_in_rank_[i * kNumRanks + dst_rank] ? 1 : 0;
                
                // Subgroup reduction sum
                count = warp_reduce_sum(count, item);

                // First thread in subgroup writes the result
                if (elect_one_sync(item))
                    channel_prefix_matrix_[dst_rank * num_channels_ + channel_id] = count;
            }
            
            // Intra-block synchronization
            item.barrier(sycl::access::fence_space::local_space);

            // Compute channel prefix sum
            if (thread_id == 0) {
                #pragma unroll
                for (int i = 1; i < num_channels_; ++i)
                    channel_prefix_matrix_[dst_rank * num_channels_ + i] += 
                        channel_prefix_matrix_[dst_rank * num_channels_ + i - 1];
            }
        }
    }

private:
    const int* num_tokens_per_rank_;
    int* moe_recv_counter_mapped_;
    const int* num_tokens_per_expert_;
    int* moe_recv_expert_counter_mapped_;
    int num_experts_;
    int num_tokens_;
    int num_channels_;
    const bool* is_token_in_rank_;
    int* channel_prefix_matrix_;
    int* rank_prefix_matrix_copy_;
    int num_memset_int_;
    int expert_alignment_;
    void** buffer_ptrs_;
    int** barrier_signal_ptrs_;
    int rank_;
};


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
                     int num_channels) {
    

    constexpr int kNumThreads = 128;
    
    // Validate parameters
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads && num_ranks <= kNumThreads);

    // Compute grid and block dimensions
    int num_blocks = 1 + num_ranks;
    sycl::range<1> global_range(num_blocks * kNumThreads);
    sycl::range<1> local_range(kNumThreads);

    // Select template instantiation by num_ranks
    #define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                          \
        case ranks: {                                                                   \
                stream.submit([&](sycl::handler& cgh) {                                \
                                                                                                            cgh.parallel_for(                                                  \
                        sycl::nd_range<1>(global_range, local_range),                 \
                        [=](sycl::nd_item<1> item) {                                  \
                            NotifyDispatchKernel<ranks> kernel(                       \
                                num_tokens_per_rank,                                   \
                                moe_recv_counter_mapped,                               \
                                num_tokens_per_expert,                                 \
                                moe_recv_expert_counter_mapped,                        \
                                num_experts,                                           \
                                num_tokens,                                            \
                                num_channels,                                          \
                                is_token_in_rank,                                      \
                                channel_prefix_matrix,                                 \
                                rank_prefix_matrix_copy,                               \
                                num_memset_int,                                        \
                                expert_alignment,                                      \
                                buffer_ptrs,                                           \
                                barrier_signal_ptrs,                                   \
                                rank);                                                   \
                            kernel(item);                                              \
                        });                                                            \
                });                                                                    \
                break;                                                                 \
        }

    switch (num_ranks) {
        NOTIFY_DISPATCH_LAUNCH_CASE(2);
        NOTIFY_DISPATCH_LAUNCH_CASE(4);
        NOTIFY_DISPATCH_LAUNCH_CASE(8);
        default:
            EP_HOST_ASSERT(false && "Unsupported number of ranks");
    }

    #undef NOTIFY_DISPATCH_LAUNCH_CASE
    
    try {
        stream.wait();
    } catch (sycl::exception const& e) {
        throw;
    } catch (std::exception const& e) {
        throw;
    }
    
    
}

}  // namespace intranode
}  // namespace deep_ep
