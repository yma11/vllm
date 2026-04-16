#!/usr/bin/env python
"""
XPU get_dispatch_layout kernel correctness test

Test three outputs of the SYCL layout kernel:
  - num_tokens_per_rank    [num_ranks]              deduplicated token count
  - num_tokens_per_expert  [num_experts]             non-deduplicated expert count
  - is_token_in_rank       [num_tokens, num_ranks]   deduplicated bool routing table

Test commands:
    mpirun -np 2 python tests/test_xpu_layout.py
    mpirun -np 2 python tests/test_xpu_layout.py --num-tokens 128 --num-topk 4 --num-experts 16
    mpirun -np 2 python tests/test_xpu_layout.py --sweep
    mpirun -np 1 python tests/test_xpu_layout.py --sweep
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

from mpi4py import MPI

os.environ['USE_XPU'] = '1'
os.environ['USE_CUDA'] = '0'


def init_dist_mpi(port: int = 29500):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    torch.xpu.set_device(rank)
    device = f'xpu:{rank}'

    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    try:
        dist.init_process_group(
            backend='xccl',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{port}',
            world_size=world_size, rank=rank)
    except Exception:
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{port}',
            world_size=world_size, rank=rank)

    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device, comm


# ─── Python reference implementation ───

def reference_layout(topk_idx_cpu, num_experts, num_ranks):
    """Compute expected results independently on CPU (source of truth)."""
    num_tokens, num_topk = topk_idx_cpu.shape
    experts_per_rank = num_experts // num_ranks

    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32)
    is_token_in_rank = torch.zeros(num_tokens, num_ranks, dtype=torch.bool)

    for i in range(num_tokens):
        for k in range(num_topk):
            e = topk_idx_cpu[i, k].item()
            if 0 <= e < num_experts:
                num_tokens_per_expert[e] += 1
                r = e // experts_per_rank
                is_token_in_rank[i, r] = True

    num_tokens_per_rank = is_token_in_rank.sum(dim=0).to(torch.int32)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


# ─── Test case generation ───

def gen_uniform_random(num_tokens, num_topk, num_experts, seed):
    """Uniform random: each topk slot independently selects a random expert."""
    torch.manual_seed(seed)
    return torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int64)


def gen_deterministic(num_tokens, num_topk, num_experts, seed):
    """Deterministic: token i's k-th topk selects expert (i+k) % num_experts."""
    topk_idx = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
    for i in range(num_tokens):
        for k in range(num_topk):
            topk_idx[i, k] = (i + k) % num_experts
    return topk_idx


def gen_same_rank_dedup(num_tokens, num_topk, num_experts, num_ranks, seed):
    """Dedup test: multiple topk slots of each token select different experts on the same rank."""
    experts_per_rank = num_experts // num_ranks
    if num_topk > experts_per_rank:
        return None  # cannot construct this case
    topk_idx = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
    for i in range(num_tokens):
        target_rank = i % num_ranks
        base = target_rank * experts_per_rank
        for k in range(num_topk):
            topk_idx[i, k] = base + (k % experts_per_rank)
    return topk_idx


def gen_same_expert(num_tokens, num_topk, num_experts, seed):
    """Repeated expert: all topk slots of each token select the same expert."""
    topk_idx = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
    for i in range(num_tokens):
        e = i % num_experts
        topk_idx[i, :] = e
    return topk_idx


def gen_with_neg1(num_tokens, num_topk, num_experts, seed):
    """With -1: odd-indexed topk slots are set to -1."""
    torch.manual_seed(seed)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int64)
    for k in range(1, num_topk, 2):
        topk_idx[:, k] = -1
    return topk_idx


def gen_all_neg1(num_tokens, num_topk, num_experts, seed):
    """All -1: every topk slot is set to -1."""
    return torch.full((num_tokens, num_topk), -1, dtype=torch.int64)


def gen_all_to_rank0(num_tokens, num_topk, num_experts, num_ranks, seed):
    """Concentrated to rank 0: all expert selections fall within rank 0's range."""
    experts_per_rank = num_experts // num_ranks
    torch.manual_seed(seed)
    return torch.randint(0, experts_per_rank, (num_tokens, num_topk), dtype=torch.int64)


def build_test_cases(num_tokens, num_topk, num_experts, num_ranks, seed):
    """Build all test cases."""
    cases = []

    cases.append(("uniform_random",
                  gen_uniform_random(num_tokens, num_topk, num_experts, seed)))

    cases.append(("deterministic",
                  gen_deterministic(num_tokens, num_topk, num_experts, seed)))

    dedup = gen_same_rank_dedup(num_tokens, num_topk, num_experts, num_ranks, seed)
    if dedup is not None:
        cases.append(("same_rank_dedup", dedup))

    if num_topk >= 2:
        cases.append(("same_expert",
                      gen_same_expert(num_tokens, num_topk, num_experts, seed)))

    if num_topk >= 2:
        cases.append(("with_neg1",
                      gen_with_neg1(num_tokens, num_topk, num_experts, seed)))

    cases.append(("all_neg1",
                  gen_all_neg1(num_tokens, num_topk, num_experts, seed)))

    cases.append(("all_to_rank0",
                  gen_all_to_rank0(num_tokens, num_topk, num_experts, num_ranks, seed)))

    if num_tokens > 0:
        cases.append(("single_token",
                      gen_uniform_random(1, num_topk, num_experts, seed + 99)))

    return cases


# ─── Verification logic ───

def verify_one(name, topk_idx_cpu, num_experts, num_ranks, buffer, device, rank):
    """Run one test case and verify."""
    import deep_ep_xpu as deep_ep

    num_tokens, num_topk = topk_idx_cpu.shape

    # Kernel execution
    topk_idx_gpu = topk_idx_cpu.to(device).to(deep_ep.topk_idx_t)
    result = buffer.get_dispatch_layout(topk_idx_gpu, num_experts)
    actual_per_rank, actual_rdma, actual_per_expert, actual_is_in_rank, _ = result
    torch.xpu.synchronize()

    # to CPU
    actual_per_rank = actual_per_rank.cpu()
    actual_per_expert = actual_per_expert.cpu()
    actual_is_in_rank = actual_is_in_rank.cpu()

    # Reference implementation
    exp_per_rank, exp_per_expert, exp_is_in_rank = reference_layout(
        topk_idx_cpu, num_experts, num_ranks)

    # Compare
    errors = []

    if not torch.equal(actual_per_expert, exp_per_expert):
        diff_mask = actual_per_expert != exp_per_expert
        diff_indices = diff_mask.nonzero(as_tuple=True)[0].tolist()
        errors.append(
            f"num_tokens_per_expert mismatch at indices {diff_indices[:10]}: "
            f"got {actual_per_expert[diff_mask].tolist()[:10]}, "
            f"expected {exp_per_expert[diff_mask].tolist()[:10]}")

    if not torch.equal(actual_is_in_rank, exp_is_in_rank):
        diff_mask = actual_is_in_rank != exp_is_in_rank
        num_diff = diff_mask.sum().item()
        errors.append(
            f"is_token_in_rank mismatch: {num_diff}/{num_tokens * num_ranks} elements differ")

    if not torch.equal(actual_per_rank, exp_per_rank):
        errors.append(
            f"num_tokens_per_rank mismatch: got {actual_per_rank.tolist()}, "
            f"expected {exp_per_rank.tolist()}")

    # Cross-consistency check
    cross_check = actual_is_in_rank.sum(dim=0).to(torch.int32)
    if not torch.equal(actual_per_rank, cross_check):
        errors.append(
            f"cross-check failed: per_rank={actual_per_rank.tolist()} vs "
            f"is_in_rank.sum(0)={cross_check.tolist()}")

    return errors


# ─── Main test ───

def run_tests(args):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import deep_ep_xpu as deep_ep

    rank, num_ranks, group, device, mpi_comm = init_dist_mpi(args.port)

    buffer = deep_ep.Buffer(
        group, int(1e8), 0,
        low_latency_mode=False,
        comm=mpi_comm)

    mpi_comm.Barrier()

    if args.sweep:
        param_grid = []
        for nt in [1, 8, 64, 256]:
            for nk in [1, 2, 4]:
                for ne in [4, 8, 16]:
                    if ne % num_ranks == 0 and ne // num_ranks >= nk:
                        param_grid.append((nt, nk, ne))
    else:
        nt, nk, ne = args.num_tokens, args.num_topk, args.num_experts
        assert ne % num_ranks == 0, \
            f"num_experts ({ne}) must be divisible by num_ranks ({num_ranks})"
        param_grid = [(nt, nk, ne)]

    total_pass = 0
    total_fail = 0

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  get_dispatch_layout correctness test")
        print(f"  num_ranks={num_ranks}, device={device}")
        print(f"  parameter combinations: {len(param_grid)}")
        print(f"{'='*60}\n")

    for nt, nk, ne in param_grid:
        cases = build_test_cases(nt, nk, ne, num_ranks, args.seed)
        # Add a num_tokens=0 edge case
        if nt > 0:
            cases.append(("empty_tokens",
                          torch.zeros(0, nk, dtype=torch.int64)))

        if rank == 0:
            print(f"--- tokens={nt}, topk={nk}, experts={ne} "
                  f"({len(cases)} cases) ---")

        for name, topk_idx_cpu in cases:
            errors = verify_one(
                name, topk_idx_cpu, ne, num_ranks, buffer, device, rank)

            if errors:
                total_fail += 1
                if rank == 0:
                    print(f"  \u2717 {name} (tokens={topk_idx_cpu.shape[0]})")
                    for e in errors:
                        print(f"      {e}")
            else:
                total_pass += 1
                if rank == 0:
                    print(f"  \u2713 {name} (tokens={topk_idx_cpu.shape[0]})")

    mpi_comm.Barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Result: {total_pass} passed, {total_fail} failed")
        print(f"{'='*60}\n")

    if total_fail > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Test XPU get_dispatch_layout kernel')
    parser.add_argument('--num-tokens', type=int, default=64)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep parameter grid automatically')
    parser.add_argument('--port', type=int, default=29500)
    args = parser.parse_args()
    run_tests(args)


if __name__ == '__main__':
    main()
