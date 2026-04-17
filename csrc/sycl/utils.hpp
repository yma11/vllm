

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


SYCL_EXTERNAL inline void get_channel_task_range(int num_tokens, int num_eus, int eu_id, 
                                          int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_eu = ceil_div(num_tokens, num_eus);
    token_start_idx = sycl::min(num_tokens_per_eu * eu_id, num_tokens);
    token_end_idx = sycl::min(token_start_idx + num_tokens_per_eu, num_tokens);
}

template <typename T>
SYCL_EXTERNAL inline T ld_nc_global(const T* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
    // IPC cross-GPU reads must bypass cache (LSC uncached)
    // Use d64 when possible to halve PCIe transactions
    static_assert(sizeof(T) % sizeof(int) == 0, "ld_nc_global requires sizeof(T) divisible by 4");
    constexpr int N = sizeof(T) / sizeof(int);
    if constexpr (N >= 2 && N % 2 == 0) {
        constexpr int N64 = N / 2;
        union { T val; int64_t i64s[N64]; } u;
        const int64_t* p = reinterpret_cast<const int64_t*>(ptr);
        #pragma unroll
        for (int i = 0; i < N64; ++i) {
            int64_t tmp;
            asm volatile(
                "lsc_load.ugm.uc.uc (M1, 32) %0:d64 flat[%1]:a64"
                : "=rw"(tmp) : "rw"(p + i)
            );
            u.i64s[i] = tmp;
        }
        return u.val;
    } else {
        union { T val; int ints[N]; } u;
        const int* p = reinterpret_cast<const int*>(ptr);
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            int tmp;
            asm volatile(
                "lsc_load.ugm.uc.uc (M1, 32) %0:d32 flat[%1]:a64"
                : "=rw"(tmp) : "rw"(p + i)
            );
            u.ints[i] = tmp;
        }
        return u.val;
    }
#else
    return *ptr;
#endif
}

template <typename T>
SYCL_EXTERNAL inline void st_na_global(T* ptr, T value) {
#ifdef __SYCL_DEVICE_ONLY__
    // IPC cross-GPU writes must bypass cache (LSC uncached)
    // Use d64 when possible to halve PCIe transactions
    static_assert(sizeof(T) % sizeof(int) == 0, "st_na_global requires sizeof(T) divisible by 4");
    constexpr int N = sizeof(T) / sizeof(int);
    if constexpr (N >= 2 && N % 2 == 0) {
        constexpr int N64 = N / 2;
        union { T val; int64_t i64s[N64]; } u;
        u.val = value;
        int64_t* p = reinterpret_cast<int64_t*>(ptr);
        #pragma unroll
        for (int i = 0; i < N64; ++i) {
            asm volatile(
                "lsc_store.ugm.uc.uc (M1, 32) flat[%0]:a64 %1:d64"
                : : "rw"(p + i), "rw"(u.i64s[i]) : "memory"
            );
        }
    } else {
        union { T val; int ints[N]; } u;
        u.val = value;
        int* p = reinterpret_cast<int*>(ptr);
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            asm volatile(
                "lsc_store.ugm.uc.uc (M1, 32) flat[%0]:a64 %1:d32"
                : : "rw"(p + i), "rw"(u.ints[i]) : "memory"
            );
        }
    }
#else
    *ptr = value;
#endif
}

// d32x4 vector store — single LSC instruction for 16 bytes per lane
template <typename T>
SYCL_EXTERNAL inline void st_na_global_v(T* ptr, T value) {
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(sizeof(T) == 16, "st_na_global_v requires sizeof(T) == 16");
    using vec4_t = typename sycl::vec<uint32_t, 4>::vector_t;
    vec4_t tmp;
    __builtin_memcpy(&tmp, &value, 16);
    auto* addr = reinterpret_cast<void*>(ptr);
    asm volatile(
        "lsc_store.ugm.uc.uc (M1, 32) flat[%0]:a64 %1:d32x4"
        : : "rw"(addr), "rw"(tmp) : "memory"
    );
#else
    *ptr = value;
#endif
}

// d32x4 vector load — single LSC instruction for 16 bytes per lane (vs 2x d64)
// Only for 16-byte types (e.g. int4 = 4x int32)
template <typename T>
SYCL_EXTERNAL inline T ld_nc_global_v(const T* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(sizeof(T) == 16, "ld_nc_global_v requires sizeof(T) == 16");
    using vec4_t = typename sycl::vec<uint32_t, 4>::vector_t;
    vec4_t tmp;
    auto* addr = reinterpret_cast<const void*>(ptr);
    asm volatile(
        "lsc_load.ugm.uc.uc (M1, 32) %0:d32x4 flat[%1]:a64"
        : "=rw"(tmp) : "rw"(addr)
    );
    T result;
    __builtin_memcpy(&result, &tmp, 16);
    return result;
#else
    return *ptr;
#endif
}

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }


template <typename T>
SYCL_EXTERNAL inline T warp_broadcast(T value, int src_lane, sycl::nd_item<1>& item) {
    auto sg = item.get_sub_group();
    return sycl::group_broadcast(sg, value, src_lane);
}
