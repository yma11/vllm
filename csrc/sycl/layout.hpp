#pragma once
#include "configs.h"
#include <sycl/sycl.hpp>
#include <cstdint>

// Note: NUM_MAX_IPC_PEERS should be defined before including this header
// It is typically defined in configs.h

namespace deep_ep {

namespace layout {

// Host function: launch SYCL kernel for dispatch layout computation
// This function computes token distribution statistics across experts and ranks
// for MoE (Mixture of Experts) models using Intel SYCL.
//
// Parameters:
//   topk_idx: Pointer to top-k expert indices [num_tokens, num_topk]
//   num_tokens_per_rank: Output array for token count per rank [num_ranks]
//   num_tokens_per_rdma_rank: Output array for token count per RDMA rank [num_rdma_ranks] (optional)
//   num_tokens_per_expert: Output array for token count per expert [num_experts]
//   is_token_in_rank: Output boolean array indicating token-rank assignment [num_tokens, num_ranks]
//   num_tokens: Total number of tokens
//   num_topk: Number of top-k experts per token
//   num_ranks: Total number of ranks
//   num_experts: Total number of experts
//   q: SYCL queue for kernel submission
void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         sycl::queue& q);

}  // namespace layout

}  // namespace deep_ep
