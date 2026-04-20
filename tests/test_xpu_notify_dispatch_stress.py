#!/usr/bin/env python
"""XPU notify_dispatch stress test

Modes:
  verify:   Functional correctness (multi-scenario x Python reference comparison)
  perf:     End-to-end performance test (Python loop + host timing)
  profile:  Capture JSON trace via torch.profiler

Commands:
    # mpirun (default, DEEPEP_LAUNCHER=mpi)
    mpirun -np 2 python tests/test_xpu_notify_dispatch_stress.py --mode verify
    mpirun -np 2 python tests/test_xpu_notify_dispatch_stress.py --mode verify \
        --num-tokens 128 --num-topk 4 --num-experts 16
    mpirun -np 2 python tests/test_xpu_notify_dispatch_stress.py --mode perf \
        --inner-repeat 100 --warmup 10
    # torchrun (set DEEPEP_LAUNCHER=torchrun)
    DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=4 tests/test_xpu_notify_dispatch_stress.py --mode verify
    DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=4 tests/test_xpu_notify_dispatch_stress.py --mode perf --inner-repeat 100
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager

LAUNCHER = os.environ.get('DEEPEP_LAUNCHER', 'mpi').lower()

if LAUNCHER == 'mpi':
    from mpi4py import MPI


# ─── Initialization ──────────────────────────────────────────


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
    return rank, world_size, group, device


def init_dist_torchrun():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.xpu.set_device(local_rank)
    device = f'xpu:{local_rank}'
    dist.init_process_group(backend='xccl')
    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device


# ─── Utility functions ────────────────────────────────────────


def get_channel_task_range(num_tokens, num_channels, channel_id):
    """Matches the kernel's get_channel_task_range logic"""
    tokens_per_channel = num_tokens // num_channels
    remainder = num_tokens % num_channels
    if channel_id < remainder:
        start = channel_id * (tokens_per_channel + 1)
        end = start + tokens_per_channel + 1
    else:
        start = remainder * (tokens_per_channel + 1) + \
            (channel_id - remainder) * tokens_per_channel
        end = start + tokens_per_channel
    return start, end


@contextmanager
def suppress_cpp_stdout():
    """Redirect fd 1 to /dev/null to suppress C++ std::cout output"""
    sys.stdout.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)


# ─── Python reference implementation ─────────────────────────────────────


def allgather_object(obj, group, num_ranks):
    """Allgather a Python object using torch.distributed (works with any launcher)"""
    result = [None] * num_ranks
    dist.all_gather_object(result, obj, group)
    return result


def reference_notify_dispatch(num_tokens_per_rank_cpu, num_tokens_per_expert_cpu,
                              is_token_in_rank_cpu, num_ranks, num_experts,
                              num_channels, expert_alignment, rank, group):
    """
    Python reference implementation. Uses MPI allgather to collect data from
    all ranks and compute expected results.

    Returns: (moe_recv_count, expert_counts, rank_prefix_col, channel_prefix)
    """
    # 1. allgather per-rank counts
    local_per_rank = num_tokens_per_rank_cpu.tolist()
    all_per_rank = allgather_object(local_per_rank, group, num_ranks)
    # all_per_rank[src][dst] = tokens from src to dst

    # 2. moe_recv_count
    moe_recv_count = sum(all_per_rank[src][rank] for src in range(num_ranks))

    # 3. Prefix sum of rank_prefix_matrix column for this rank
    rank_prefix_col = []
    prefix = 0
    for i in range(num_ranks):
        prefix += all_per_rank[i][rank]
        rank_prefix_col.append(prefix)

    # 4. expert_counts (with alignment)
    epr = num_experts // num_ranks
    local_per_expert = num_tokens_per_expert_cpu.tolist()
    all_per_expert = allgather_object(local_per_expert, group, num_ranks)

    expert_counts = []
    for e in range(epr):
        global_e = rank * epr + e
        total = sum(all_per_expert[src][global_e] for src in range(num_ranks))
        aligned = (total + expert_alignment - 1) // expert_alignment * expert_alignment
        expert_counts.append(aligned)

    # 5. channel_prefix_matrix (computed locally)
    num_tokens = is_token_in_rank_cpu.shape[0]
    channel_prefix = torch.zeros(num_ranks, num_channels, dtype=torch.int32)

    if num_tokens > 0:
        for dst in range(num_ranks):
            for ch in range(num_channels):
                start, end = get_channel_task_range(num_tokens, num_channels, ch)
                count = int(is_token_in_rank_cpu[start:end, dst].sum())
                channel_prefix[dst, ch] = count
            for ch in range(1, num_channels):
                channel_prefix[dst, ch] += channel_prefix[dst, ch - 1]

    return moe_recv_count, expert_counts, rank_prefix_col, channel_prefix


# ─── Test case generation ─────────────────────────────────────


def generate_topk(scenario, num_tokens, num_topk, num_experts, num_ranks, rank, seed):
    """Generate topk_idx [num_tokens, num_topk] int64"""
    epr = num_experts // num_ranks
    torch.manual_seed(seed + rank)

    if scenario == 'uniform_random':
        return torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int64)

    elif scenario == 'deterministic':
        topk = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
        for i in range(num_tokens):
            for k in range(num_topk):
                topk[i, k] = (i + k + rank) % num_experts
        return topk

    elif scenario == 'all_to_rank0':
        return torch.randint(0, max(epr, 1), (num_tokens, num_topk), dtype=torch.int64)

    elif scenario == 'all_to_self':
        base = rank * epr
        return torch.randint(base, base + max(epr, 1), (num_tokens, num_topk), dtype=torch.int64)

    elif scenario == 'single_token':
        return torch.randint(0, num_experts, (1, num_topk), dtype=torch.int64)

    elif scenario == 'empty':
        return torch.zeros(0, num_topk, dtype=torch.int64)

    elif scenario == 'same_rank_multi_expert':
        if num_topk > epr:
            return None
        topk = torch.zeros(num_tokens, num_topk, dtype=torch.int64)
        for i in range(num_tokens):
            target_rank = (i + rank) % num_ranks
            base = target_rank * epr
            for k in range(num_topk):
                topk[i, k] = base + (k % epr)
        return topk

    return None


def build_scenarios(num_tokens, num_topk, num_experts, num_ranks, rank, seed):
    """Build all test scenarios"""
    names = [
        'uniform_random', 'deterministic', 'all_to_rank0', 'all_to_self',
        'single_token', 'empty',
    ]
    if num_topk >= 2:
        names.append('same_rank_multi_expert')

    cases = []
    for s in names:
        topk = generate_topk(s, num_tokens, num_topk, num_experts, num_ranks, rank, seed)
        if topk is not None:
            cases.append((s, topk))
    return cases


# ─── Verification logic ────────────────────────────────────────


def verify_one(name, topk_idx_cpu, num_experts, num_channels, expert_alignment,
               buffer, device, rank, num_ranks, group):
    """Run one test case, return list of errors (empty = pass)"""
    import deep_ep_xpu as deep_ep

    actual_tokens = topk_idx_cpu.shape[0]

    # Generate inputs via get_dispatch_layout
    topk_idx_gpu = topk_idx_cpu.to(device).to(deep_ep.topk_idx_t)
    per_rank, _, per_expert, is_in_rank, _ = buffer.get_dispatch_layout(
        topk_idx_gpu, num_experts)
    torch.xpu.synchronize()

    # Call test_notify_dispatch
    moe_recv_count, expert_counts, rank_prefix_matrix, channel_prefix_matrix = \
        buffer.runtime.test_notify_dispatch(
            per_rank, per_expert, is_in_rank,
            actual_tokens, num_experts, num_channels, expert_alignment)
    torch.xpu.synchronize()

    # Python reference
    exp_recv, exp_experts, exp_prefix_col, exp_channel = reference_notify_dispatch(
        per_rank.cpu(), per_expert.cpu(), is_in_rank.cpu(),
        num_ranks, num_experts, num_channels, expert_alignment, rank, group)

    errors = []

    # 1. moe_recv_count
    if moe_recv_count != exp_recv:
        errors.append(f"moe_recv_count: got {moe_recv_count}, expected {exp_recv}")

    # 2. expert_counts
    if expert_counts != exp_experts:
        errors.append(f"expert_counts: got {expert_counts}, expected {exp_experts}")

    # 3. rank_prefix_matrix[:, rank]
    rpm_cpu = rank_prefix_matrix.cpu()
    for i in range(num_ranks):
        actual_val = rpm_cpu[i, rank].item()
        expected_val = exp_prefix_col[i]
        if actual_val != expected_val:
            errors.append(
                f"rank_prefix[{i},{rank}]: got {actual_val}, expected {expected_val}")

    # Consistency: last row == moe_recv_count
    if exp_prefix_col and exp_prefix_col[-1] != exp_recv:
        errors.append(
            f"consistency: prefix_col[-1]={exp_prefix_col[-1]} != recv={exp_recv}")

    # 4. channel_prefix_matrix
    cpm_cpu = channel_prefix_matrix.cpu()
    for dst in range(num_ranks):
        for ch in range(num_channels):
            actual_val = cpm_cpu[dst, ch].item()
            expected_val = exp_channel[dst, ch].item()
            if actual_val != expected_val:
                errors.append(
                    f"channel_prefix[{dst},{ch}]: got {actual_val}, expected {expected_val}")
                break  # Avoid flooding output

    # Consistency: last column == num_tokens_per_rank
    per_rank_cpu = per_rank.cpu()
    if actual_tokens > 0 and num_channels > 0:
        for dst in range(num_ranks):
            last_ch = cpm_cpu[dst, num_channels - 1].item()
            expected_total = per_rank_cpu[dst].item()
            if last_ch != expected_total:
                errors.append(
                    f"channel_prefix[{dst},last]={last_ch} != per_rank[{dst}]={expected_total}")

    return errors


# ─── Main flow ────────────────────────────────────────────────


def run_tests(args):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import deep_ep_xpu as deep_ep

    if LAUNCHER == 'torchrun':
        rank, num_ranks, group, device = init_dist_torchrun()
    else:
        rank, num_ranks, group, device = init_dist_mpi(args.port)

    buffer = deep_ep.Buffer(
        group, int(1e8), 0,
        low_latency_mode=False)

    dist.barrier(group=group)

    mode = args.mode
    num_tokens = args.num_tokens
    num_topk = args.num_topk
    num_experts = args.num_experts
    num_channels = args.num_channels
    expert_alignment = args.expert_alignment

    assert num_experts % num_ranks == 0, \
        f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  notify_dispatch stress test [{mode}]")
        print(f"  num_ranks={num_ranks}, device={device}")
        print(f"  tokens={num_tokens}, topk={num_topk}, experts={num_experts}")
        print(f"  channels={num_channels}, alignment={expert_alignment}")
        print(f"{'='*60}\n")

    # ──────── verify ────────
    if mode == 'verify':
        cases = build_scenarios(
            num_tokens, num_topk, num_experts, num_ranks, rank, args.seed)
        total_pass = 0
        total_fail = 0

        for name, topk_cpu in cases:
            dist.barrier(group=group)
            errors = verify_one(
                name, topk_cpu, num_experts, num_channels,
                expert_alignment, buffer, device, rank, num_ranks, group)
            if errors:
                total_fail += 1
                if rank == 0:
                    print(f"  FAIL {name} (tokens={topk_cpu.shape[0]})")
                    for e in errors:
                        print(f"      {e}")
            else:
                total_pass += 1
                if rank == 0:
                    print(f"  PASS {name} (tokens={topk_cpu.shape[0]})")

        dist.barrier(group=group)
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  Result: {total_pass} passed, {total_fail} failed")
            print(f"{'='*60}\n")
        if total_fail > 0:
            sys.exit(1)

    # ──────── perf ────────
    elif mode == 'perf':
        inner = args.inner_repeat
        warmup = args.warmup

        torch.manual_seed(args.seed + rank)
        topk_idx = torch.randint(
            0, num_experts, (num_tokens, num_topk),
            dtype=torch.int64, device=device).to(deep_ep.topk_idx_t)
        per_rank, _, per_expert, is_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts)
        torch.xpu.synchronize()

        # warmup
        if warmup > 0:
            if rank == 0:
                print(f"[Warmup] {warmup} iterations...")
            with suppress_cpp_stdout():
                for _ in range(warmup):
                    buffer.runtime.test_notify_dispatch(
                        per_rank, per_expert, is_in_rank,
                        num_tokens, num_experts, num_channels, expert_alignment)
            torch.xpu.synchronize()
            dist.barrier(group=group)

        # timed run
        dist.barrier(group=group)
        torch.xpu.synchronize()
        t0 = time.time()
        with suppress_cpp_stdout():
            for _ in range(inner):
                buffer.runtime.test_notify_dispatch(
                    per_rank, per_expert, is_in_rank,
                    num_tokens, num_experts, num_channels, expert_alignment)
        torch.xpu.synchronize()
        elapsed = time.time() - t0
        dist.barrier(group=group)

        if rank == 0:
            avg_us = elapsed * 1e6 / inner
            print(f"\n{'='*60}")
            print(f"  PERF: {inner} iterations in {elapsed*1000:.1f} ms")
            print(f"  Avg: {avg_us:.1f} us/call")
            print(f"  Note: includes C++ debug print overhead (fd suppressed)")
            print(f"{'='*60}\n")

    # ──────── profile ────────
    elif mode == 'profile':
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

        inner = args.inner_repeat
        trace_dir = args.trace_dir
        os.makedirs(trace_dir, exist_ok=True)

        torch.manual_seed(args.seed + rank)
        topk_idx = torch.randint(
            0, num_experts, (num_tokens, num_topk),
            dtype=torch.int64, device=device).to(deep_ep.topk_idx_t)
        per_rank, _, per_expert, is_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts)
        torch.xpu.synchronize()

        activities = [ProfilerActivity.CPU]
        if hasattr(ProfilerActivity, 'XPU'):
            activities.append(ProfilerActivity.XPU)

        with profile(
            activities=activities,
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=False, with_stack=False,
        ) as prof:
            for _ in range(inner):
                with suppress_cpp_stdout():
                    buffer.runtime.test_notify_dispatch(
                        per_rank, per_expert, is_in_rank,
                        num_tokens, num_experts, num_channels, expert_alignment)
                prof.step()

        dist.barrier(group=group)
        if rank == 0:
            json_files = [f for f in os.listdir(trace_dir) if f.endswith('.json')]
            print(f"\n{'='*60}")
            print(f"  PROFILE: {inner} iterations, {len(json_files)} trace files")
            print(f"  Output: {trace_dir}/")
            print(f"{'='*60}\n")

    dist.barrier(group=group)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description='XPU notify_dispatch stress test')
    parser.add_argument('--mode', choices=['verify', 'perf', 'profile'],
                        default='verify')
    parser.add_argument('--num-tokens', type=int, default=64)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-channels', type=int, default=4)
    parser.add_argument('--expert-alignment', type=int, default=1)
    parser.add_argument('--inner-repeat', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trace-dir', type=str, default='./nd_traces')
    parser.add_argument('--port', type=int, default=29500)
    args = parser.parse_args()
    run_tests(args)


if __name__ == '__main__':
    main()
