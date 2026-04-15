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

}  // namespace intranode
}  // namespace deep_ep
