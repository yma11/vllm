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


}  // namespace deep_ep
