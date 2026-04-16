

#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include "configs.h"

inline int ld_volatile_global(const int* addr) {
#ifdef __SYCL_DEVICE_ONLY__
    int result;
    asm volatile (
        "lsc_load.ugm.uc.uc (M1, 32) %0:d32 flat[%1]:a64"
        : "=rw"(result) : "rw"(addr)
    );
    return result;
#else
    return *static_cast<const volatile int*>(addr);
#endif
}

inline int64_t ld_volatile_global(const int64_t* addr) {
#ifdef __SYCL_DEVICE_ONLY__
    int64_t result;
    asm volatile (
        "lsc_load.ugm.uc.uc (M1, 32) %0:d64 flat[%1]:a64"
        : "=rw"(result) : "rw"(addr)
    );
    return result;
#else
    return *static_cast<const volatile int64_t*>(addr);
#endif
}


inline void st_volatile_global(int* addr, int value) {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile (
        "lsc_store.ugm.uc.uc (M1, 32) flat[%0]:a64 %1:d32"
        : : "rw"(addr), "rw"(value) : "memory"
    );
#else
    *static_cast<volatile int*>(addr) = value;
#endif
}

inline void st_volatile_global(int64_t* addr, int64_t value) {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile (
        "lsc_store.ugm.uc.uc (M1, 32) flat[%0]:a64 %1:d64"
        : : "rw"(addr), "rw"(value) : "memory"
    );
#else
    *static_cast<volatile int64_t*>(addr) = value;
#endif
}

SYCL_EXTERNAL inline void memory_fence_system() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

SYCL_EXTERNAL inline void memory_fence_device() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

SYCL_EXTERNAL inline void memory_fence_workgroup() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

template <int kNumRanks, bool kSyncOnly = false>
SYCL_EXTERNAL inline void barrier_block_bypass(int** barrier_signal_ptrs, int rank,
                                                sycl::nd_item<1>& item
) {
    auto thread_id = static_cast<int>(item.get_local_id(0));

    if constexpr (!kSyncOnly) {
        memory_fence_system();
        item.barrier(sycl::access::fence_space::local_space);
    }

    int next_epoch = 0;

    // Step 1: write to our own slot (ptrs[rank][thread_id], exclusively owned by this rank)
    if (thread_id < kNumRanks) {
        int* my_slot = barrier_signal_ptrs[rank] + thread_id;
        int cur_epoch = ld_volatile_global(my_slot);
        next_epoch = cur_epoch + 1;
        st_volatile_global(my_slot, next_epoch);
    }

    // Ensure the flag write is visible to remote GPUs (via PCIe IPC mapping)
    memory_fence_system();

    // Step 2: poll peer slot (ptrs[thread_id][rank], written by rank=thread_id)
    if (thread_id < kNumRanks) {
        int* peer_slot = barrier_signal_ptrs[thread_id] + rank;
        while (ld_volatile_global(peer_slot) < next_epoch) {
        }
    }

    item.barrier(sycl::access::fence_space::local_space);
}

template <typename T>
SYCL_EXTERNAL inline T warp_reduce_sum(T value, sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return sycl::reduce_over_group(sg, value, sycl::plus<T>());
}

template <typename T>
SYCL_EXTERNAL inline T warp_reduce_max(T value, sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return sycl::reduce_over_group(sg, value, sycl::maximum<T>());
}

template <typename T>
SYCL_EXTERNAL inline T warp_reduce_min(T value, sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return sycl::reduce_over_group(sg, value, sycl::minimum<T>());
}

SYCL_EXTERNAL inline bool elect_one_sync(sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return sg.get_local_linear_id() == 0;
}

SYCL_EXTERNAL inline int get_lane_id(sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return static_cast<int>(sg.get_local_linear_id());
}

template <typename dtype_t>
inline constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}


SYCL_EXTERNAL inline void get_channel_task_range(int num_tokens, int num_sms, int sm_id, 
                                          int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = sycl::min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = sycl::min(token_start_idx + num_tokens_per_sm, num_tokens);
}