#pragma once

#include <cstdint>

namespace deep_ep {

// ============================================================================
// Buffer wrapper class for SYCL (similar to CUDA buffer.cuh)
// Manages global memory pointers and offsets
// ============================================================================

template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    int64_t total_bytes;

    Buffer() : ptr(nullptr), total_bytes(0) {}

    // Constructor: allocate a memory region from the global pointer
    // gbl_ptr: in/out parameter, advanced to the next available position
    // num_elems: number of elements
    // offset: starting offset (in elements)
    Buffer(void*& gbl_ptr, int64_t num_elems, int64_t offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    // Also advance another global pointer
    Buffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    // Get underlying buffer pointer
    dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }

    // Get underlying buffer pointer (const version)
    const dtype_t* buffer() const { return reinterpret_cast<const dtype_t*>(ptr); }

    // Array subscript access
    dtype_t& operator[](int64_t idx) { return buffer()[idx]; }
    const dtype_t& operator[](int64_t idx) const { return buffer()[idx]; }
};

// ============================================================================
// AsymBuffer - Asymmetric buffer (for multi-rank scenarios)
// ============================================================================

template <typename dtype_t, int kNumRanks = 1>
struct AsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    int64_t num_bytes;

public:
    int64_t total_bytes;

    // Single-rank constructor
    AsymBuffer(void*& gbl_ptr, int64_t num_elems, int num_ranks, int eu_id = 0, int num_eus = 1, int64_t offset = 0) {
        static_assert(kNumRanks == 1, "This constructor is for single rank only");
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_eus;
        ptrs[0] = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * eu_id + num_bytes * offset;
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    // Multi-rank constructor
    AsymBuffer(void** gbl_ptrs, int64_t num_elems, int num_ranks, int eu_id = 0, int num_eus = 1, int64_t offset = 0) {
        static_assert(kNumRanks > 1, "This constructor is for multiple ranks");
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_eus;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * eu_id + num_bytes * offset;
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    // Advance by offset
    void advance(int64_t shift) {
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    // Also advance another global pointer
    AsymBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    // Also advance multiple global pointers
    template <int kNumAlsoRanks>
    AsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    // Get buffer pointer (single-rank case)
    dtype_t* buffer(int idx = 0) {
        static_assert(kNumRanks == 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    // Get buffer pointer for a specific rank (multi-rank case)
    dtype_t* buffer_by(int rank_idx, int idx = 0) {
        static_assert(kNumRanks > 1, "`buffer_by` is only available for multiple rank case");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

}  // namespace deep_ep
