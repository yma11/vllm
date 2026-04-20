#!/usr/bin/env python
"""
XPU Intranode Combine Stress Test

Supports two modes (--mode verify | perf) and two dtypes (--dtype int32 | bfloat16).

Dtype / mode compatibility:
  int32:    supports both verify and perf modes
  bfloat16: supports perf mode only (limited-precision arithmetic precludes exact verification)

verify mode (int32 only) — correctness check via a full dispatch -> combine pipeline:
  x[i,:] = rank * 1000000 + i
  Expected: combined_x[i,:] == original_value * num_copies[i]  (exact integer equality)

perf mode — latency and bandwidth benchmark (no verification):
  Reports per-iteration combine latency (ms) and algorithmic bandwidth (GB/s).

Example commands:
    mpirun -np 4 python tests/test_xpu_combine_stress.py --dtype int32 --mode verify
    mpirun -np 4 python tests/test_xpu_combine_stress.py --dtype int32 --mode perf --num-tokens 1024 --repeat 20
    mpirun -np 4 python tests/test_xpu_combine_stress.py --dtype bfloat16 --mode perf --num-tokens 4096 --repeat 20
"""

import argparse
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.distributed as dist

os.environ['USE_XPU'] = '1'
os.environ['USE_CUDA'] = '0'

from mpi4py import MPI

DTYPE_MAP = {
    'int32':    torch.int32,
    'bfloat16': torch.bfloat16,
}


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

    dist.init_process_group(
        backend='xccl',
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{port}',
        world_size=world_size,
        rank=rank
    )

    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device


def build_deterministic_routing(num_tokens, num_topk, num_experts, num_ranks, device):
    experts_per_rank = num_experts // num_ranks

    topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device=device)
    for i in range(num_tokens):
        for k in range(num_topk):
            target_rank = (i + k) % num_ranks
            topk_idx[i, k] = target_rank * experts_per_rank

    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device)

    is_token_in_rank = torch.zeros(num_tokens, num_ranks, dtype=torch.bool, device=device)
    for i in range(num_tokens):
        seen = set()
        for k in range(num_topk):
            r = (i + k) % num_ranks
            if r not in seen:
                is_token_in_rank[i, r] = True
                seen.add(r)

    num_tokens_per_rank = is_token_in_rank.sum(dim=0).to(torch.int32)

    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        for k in range(num_topk):
            idx = topk_idx[i, k].item()
            num_tokens_per_expert[idx] += 1

    return topk_idx, topk_weights, is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert


def build_input_tensor(num_tokens, hidden, rank, device, dtype):
    if dtype == torch.int32:
        assert num_tokens < 1000000, f"num_tokens must be < 1000000, got {num_tokens}"
        values = rank * 1000000 + torch.arange(num_tokens, dtype=torch.int32, device=device)
    else:
        # bfloat16: values 1..32 are exactly representable; product with num_copies<=8 stays <=256
        values = ((torch.arange(num_tokens, device=device) % 32) + 1).to(dtype)
    return values.unsqueeze(1).expand(num_tokens, hidden).contiguous()


def verify_combine(combined_x, is_token_in_rank, rank, num_tokens, hidden, dtype, verbose=False):
    """Verify combine output for both int32 and bfloat16 dtypes."""
    errors = []

    if combined_x.shape[0] != num_tokens:
        errors.append(f"Shape mismatch: expected {num_tokens} tokens, got {combined_x.shape[0]}")
        return errors

    num_copies = is_token_in_rank.sum(dim=1)  # shape: [num_tokens]

    for i in range(num_tokens):
        nc = num_copies[i].item()
        if dtype == torch.int32:
            original_val = rank * 1000000 + i
            expected_val = original_val * nc
            tol = 0
        else:
            original_val = float(i % 32 + 1)
            expected_val = original_val * nc
            tol = 1e-2

        actual_val = combined_x[i, 0].item()
        if abs(actual_val - expected_val) > tol:
            errors.append(
                f"Token {i}: expected {expected_val} (={original_val}*{nc}), got {actual_val}")
            if len(errors) > 20:
                errors.append("... too many errors, stopping")
                return errors
            continue

        row = combined_x[i, :].float()
        if (row - expected_val).abs().max().item() > tol:
            mismatches = ((row - expected_val).abs() > tol).sum().item()
            errors.append(
                f"Token {i}: {mismatches}/{hidden} elements differ (expected {expected_val})")
            if len(errors) > 20:
                errors.append("... too many errors, stopping")
                return errors

        if verbose and i < 3:
            print(f"  [rank {rank}] token {i}: val={actual_val} "
                  f"(original={original_val} * {nc} copies) OK", flush=True)

    return errors


def run_iteration(args, dtype, buffer, topk_idx, topk_weights, is_token_in_rank,
                  num_tokens_per_rank, num_tokens_per_expert, rank, device):
    import deep_ep_xpu as deep_ep

    config = deep_ep.Config(args.num_sms, 256, 512)
    x = build_input_tensor(args.num_tokens, args.hidden, rank, device, dtype)

    torch.xpu.synchronize()
    t0 = time.perf_counter()

    recv_x, recv_topk_idx, recv_topk_weights, recv_expert_list, handle, event = \
        buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=config,
        )
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    combined_x, combined_topk_weights, combine_event = buffer.combine(
        x=recv_x,
        handle=handle,
        topk_weights=recv_topk_weights,
        config=config,
    )
    torch.xpu.synchronize()
    t2 = time.perf_counter()

    dispatch_ms = (t1 - t0) * 1000
    combine_ms  = (t2 - t1) * 1000
    return combined_x, is_token_in_rank, dispatch_ms, combine_ms


def main():
    parser = argparse.ArgumentParser(description='Combine stress test')
    parser.add_argument('--num-tokens', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=128)
    parser.add_argument('--num-sms', type=int, default=4)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--max-errors', type=int, default=20)
    parser.add_argument('--dtype', choices=['int32', 'bfloat16'], default='int32',
                        help='Input tensor dtype (bfloat16 supports perf mode only)')
    parser.add_argument('--mode', choices=['verify', 'perf'], default='verify',
                        help='verify: correctness check (int32 only); perf: latency/bandwidth benchmark')
    args = parser.parse_args()

    if args.dtype == 'bfloat16' and args.mode == 'verify':
        parser.error('bfloat16 does not support verify mode; use --mode perf')

    import deep_ep_xpu as deep_ep

    dtype = DTYPE_MAP[args.dtype]
    rank, num_ranks, group, device = init_dist_mpi(port=args.port)
    comm = MPI.COMM_WORLD

    assert args.num_experts % num_ranks == 0

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    if rank == 0:
        print(f'[init] {num_ranks} ranks, device={device}', flush=True)
        print(f'[config] num_tokens={args.num_tokens}, hidden={args.hidden}, '
              f'num_topk={args.num_topk}, num_experts={args.num_experts}, '
              f'num_sms={args.num_sms}, repeat={args.repeat}, '
              f'dtype={args.dtype}, mode={args.mode}', flush=True)

    buffer = deep_ep.Buffer(group, int(1e9), 0, low_latency_mode=False)

    topk_idx, topk_weights, is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert = \
        build_deterministic_routing(args.num_tokens, args.num_topk, args.num_experts, num_ranks, device)

    if args.warmup > 0:
        if rank == 0:
            print(f'\n[warmup] {args.warmup} iterations...', flush=True)
        for _ in range(args.warmup):
            run_iteration(args, dtype, buffer, topk_idx, topk_weights, is_token_in_rank,
                          num_tokens_per_rank, num_tokens_per_expert, rank, device)
        if rank == 0:
            print('[warmup] done', flush=True)

    total_errors = 0
    dispatch_times = []
    combine_times  = []
    all_passed_local = True

    for iteration in range(args.repeat):
        combined_x, is_tok_in_rank, d_ms, c_ms = run_iteration(
            args, dtype, buffer, topk_idx, topk_weights, is_token_in_rank,
            num_tokens_per_rank, num_tokens_per_expert, rank, device)
        dispatch_times.append(d_ms)
        combine_times.append(c_ms)

        # Algorithmic bandwidth: num_tokens * hidden * dtype_bytes (combine output) / time
        alg_bw = args.num_tokens * args.hidden * dtype_bytes / (c_ms * 1e-3) / 1e9

        if args.mode == 'verify':
            verbose = (iteration == 0 and rank == 0)
            errs = verify_combine(combined_x, is_tok_in_rank, rank,
                                  args.num_tokens, args.hidden, dtype, verbose=verbose)
            if errs:
                all_passed_local = False
                total_errors += len(errs)
                for e in errs[:args.max_errors]:
                    print(f'[rank {rank}] iter {iteration} ERROR: {e}', flush=True)
            else:
                print(f'[rank {rank}] iter {iteration} OK '
                      f'(dispatch={d_ms:.3f}ms combine={c_ms:.3f}ms bw={alg_bw:.2f}GB/s)',
                      flush=True)
        else:  # perf
            print(f'[rank {rank}] iter {iteration} '
                  f'dispatch={d_ms:.3f}ms combine={c_ms:.3f}ms bw={alg_bw:.2f}GB/s',
                  flush=True)

    torch.xpu.synchronize()
    comm.Barrier()

    if combine_times:
        avg_c  = sum(combine_times) / len(combine_times)
        min_c  = min(combine_times)
        max_c  = max(combine_times)
        avg_bw = args.num_tokens * args.hidden * dtype_bytes / (avg_c * 1e-3) / 1e9
        print(f'[rank {rank}] combine summary: '
              f'avg={avg_c:.3f}ms min={min_c:.3f}ms max={max_c:.3f}ms '
              f'avg_bw={avg_bw:.2f}GB/s errors={total_errors}', flush=True)

    if args.mode == 'verify':
        all_passed = comm.allreduce(all_passed_local, op=MPI.LAND)
        if rank == 0:
            if all_passed:
                print('\n========== ALL PASSED ==========\n', flush=True)
            else:
                print('\n========== FAILED ==========\n', flush=True)
    else:
        all_passed = True

    try:
        dist.destroy_process_group()
    except Exception:
        pass

    comm.Barrier()
    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
