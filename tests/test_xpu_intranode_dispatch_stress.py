#!/usr/bin/env python
"""
XPU Intranode Dispatch Stress Test (int32)

Uses int32 data type to avoid bfloat16 precision issues, enabling exact dispatch verification.
Encoding: each token's hidden dimensions are filled with rank * 1000000 + token_id.
After dispatch, (src_rank, token_id) can be exactly decoded with zero precision loss.

Usage:
    # mpirun (default, DEEPEP_LAUNCHER=mpi)
    mpirun -np 2 python tests/test_xpu_intranode_dispatch_stress.py
    mpirun -np 4 python tests/test_xpu_intranode_dispatch_stress.py --num-tokens 32
    # torchrun (set DEEPEP_LAUNCHER=torchrun)
    DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=4 tests/test_xpu_intranode_dispatch_stress.py
    DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=4 tests/test_xpu_intranode_dispatch_stress.py --num-tokens 32 --repeat 10
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

LAUNCHER = os.environ.get('DEEPEP_LAUNCHER', 'mpi').lower()

if LAUNCHER == 'mpi':
    from mpi4py import MPI


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


def init_dist_torchrun():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.xpu.set_device(local_rank)
    device = f'xpu:{local_rank}'
    dist.init_process_group(backend='xccl')
    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device


def build_deterministic_routing(num_tokens, num_topk, num_experts, num_ranks, device):
    """
    Deterministic routing: token i's k-th expert selects the first expert on rank (i + k) % num_ranks.
    """
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


def build_encoded_data_int32(num_tokens, hidden, rank, device):
    """
    int32 encoding: value = rank * 1000000 + token_id.
    int32 range [-2^31, 2^31-1] supports up to 2000 ranks x 1000000 tokens.
    """
    assert num_tokens < 1000000, f"num_tokens must be < 1000000, got {num_tokens}"
    values = rank * 1000000 + torch.arange(num_tokens, dtype=torch.int32, device=device)
    x = values.unsqueeze(1).expand(num_tokens, hidden).contiguous()
    return x


def verify_dispatch_int32(recv_x, rank_prefix_matrix, rank, num_ranks,
                          all_is_token_in_rank, all_num_tokens_per_rank,
                          verbose=False):
    """
    Per-token exact verification of dispatch results (int32, zero precision loss).
    """
    errors = []
    detailed_prints = 0  # Limit number of detailed diagnostic prints

    expected_total = sum(t[rank].item() for t in all_num_tokens_per_rank)
    actual_total = recv_x.shape[0]
    if expected_total != actual_total:
        errors.append(f"Total recv count mismatch: expected {expected_total}, got {actual_total}")
        return errors

    check_start = 0

    for src_rank in range(num_ranks):
        check_end = rank_prefix_matrix[src_rank][rank].item()
        segment_size = check_end - check_start

        expected_segment = all_num_tokens_per_rank[src_rank][rank].item()
        if segment_size != expected_segment:
            errors.append(f"Segment from rank {src_rank}: expected {expected_segment} tokens, "
                          f"got {segment_size}")

        seen_tokens = set()
        for idx in range(check_start, check_end):
            val = recv_x[idx, 0].item()
            decoded_rank = val // 1000000
            decoded_token = val % 1000000

            if verbose and idx < check_start + 3:
                print(f"  [rank {rank}] idx={idx} val={val} => src_rank={decoded_rank} token={decoded_token}",
                      flush=True)

            if decoded_rank != src_rank:
                errors.append(f"Token at idx {idx}: value {val} decodes to rank {decoded_rank}, "
                              f"expected src_rank {src_rank}")
                continue

            num_tokens_src = all_is_token_in_rank[src_rank].shape[0]
            if decoded_token < 0 or decoded_token >= num_tokens_src:
                errors.append(f"Token at idx {idx}: value {val} decodes to token_id {decoded_token}, "
                              f"out of range [0, {num_tokens_src})")
                continue

            if not all_is_token_in_rank[src_rank][decoded_token, rank].item():
                errors.append(f"Token {decoded_token} from rank {src_rank} should NOT be sent "
                              f"to rank {rank}")

            # Verify all elements in the row are identical (int32 exact comparison)
            row = recv_x[idx, :]
            if not (row == val).all():
                mismatches = (row != val).sum().item()
                total_elems = row.shape[0]
                errors.append(f"Token at idx {idx}: {mismatches}/{total_elems} elements differ "
                              f"(data corruption, expected {val})")
                
                # === Detailed diagnostics: compare actual vs expected per int4 (4 x int32) ===
                if detailed_prints < 3:
                    detailed_prints += 1
                    # int4 = 16 bytes = 4 x int32, UNROLLED_WARP_COPY copies in int4 units
                    num_int4 = total_elems // 4
                    expected_val = val
                    print(f"  [rank {rank}] idx={idx}: expected_val={expected_val}, total_int4={num_int4}", flush=True)
                    
                    # Check whether each int4 block is fully correct
                    row_reshaped = row.reshape(num_int4, 4)
                    expected_row = torch.full_like(row, expected_val)
                    expected_reshaped = expected_row.reshape(num_int4, 4)
                    int4_correct = (row_reshaped == expected_reshaped).all(dim=1)
                    
                    # Print the first 10 incorrect int4 blocks (position + actual vs expected)
                    wrong_int4_indices = torch.where(~int4_correct)[0]
                    correct_int4_indices = torch.where(int4_correct)[0]
                    print(f"  [rank {rank}] idx={idx}: correct_int4={int4_correct.sum().item()}/{num_int4}, "
                          f"wrong_int4={len(wrong_int4_indices)}/{num_int4}", flush=True)
                    
                    if len(correct_int4_indices) > 0:
                        print(f"  [rank {rank}] idx={idx}: first correct int4 positions: "
                              f"{correct_int4_indices[:10].tolist()}", flush=True)
                    
                    num_show = min(10, len(wrong_int4_indices))
                    for wi in range(num_show):
                        int4_pos = wrong_int4_indices[wi].item()
                        elem_start = int4_pos * 4
                        actual_vals = row[elem_start:elem_start+4].tolist()
                        print(f"  [rank {rank}] idx={idx}: WRONG int4[{int4_pos}] (elem[{elem_start}:{elem_start+4}]): "
                              f"actual={actual_vals}, expected=[{expected_val}]*4", flush=True)
                    
                    if len(wrong_int4_indices) > num_show:
                        print(f"  [rank {rank}] idx={idx}: ... and {len(wrong_int4_indices)-num_show} more wrong int4 blocks", flush=True)
                    
                    # Summarize wrong value distribution
                    wrong_values = row[row != expected_val]
                    num_zeros = (wrong_values == 0).sum().item()
                    unique_wrong = torch.unique(wrong_values)
                    print(f"  [rank {rank}] idx={idx}: wrong value summary: "
                          f"num_zeros={num_zeros}/{len(wrong_values)}, "
                          f"unique_count={len(unique_wrong)}, "
                          f"unique_vals={unique_wrong[:15].tolist()}"
                          f"{'...' if len(unique_wrong) > 15 else ''}",
                          flush=True)

            if decoded_token in seen_tokens:
                errors.append(f"Token {decoded_token} from rank {src_rank} received twice")
            seen_tokens.add(decoded_token)

        check_start = check_end

    return errors


def run_dispatch_test(args, rank, num_ranks, group, device, buffer, iteration,
                      topk_idx, topk_weights, is_token_in_rank,
                      num_tokens_per_rank, num_tokens_per_expert,
                      all_num_tokens_per_rank, all_is_token_in_rank):
    import deep_ep_xpu as deep_ep

    num_tokens = args.num_tokens
    hidden = args.hidden

    x = build_encoded_data_int32(num_tokens, hidden, rank, device)
    config = deep_ep.Config(args.num_eus, 256, 512)

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
    elapsed_ms = (t1 - t0) * 1000

    verbose = (iteration == 0 and rank == 0)
    rank_prefix_matrix = handle[0]

    dispatch_errors = verify_dispatch_int32(
        recv_x, rank_prefix_matrix, rank, num_ranks,
        all_is_token_in_rank, all_num_tokens_per_rank,
        verbose=verbose,
    )

    return dispatch_errors, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description='Dispatch stress test (int32)')
    parser.add_argument('--num-tokens', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=128)
    parser.add_argument('--num-eus', type=int, default=4)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of dispatch iterations')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--max-errors', type=int, default=20,
                        help='Max errors to print per iteration')
    parser.add_argument('--profile', action='store_true',
                        help='Run with torch profiler to measure kernel/host time')
    args = parser.parse_args()

    import deep_ep_xpu as deep_ep

    if LAUNCHER == 'torchrun':
        rank, num_ranks, group, device = init_dist_torchrun()
    else:
        rank, num_ranks, group, device = init_dist_mpi(port=args.port)

    assert args.num_experts % num_ranks == 0, \
        f"num_experts ({args.num_experts}) must be divisible by num_ranks ({num_ranks})"

    if rank == 0:
        print(f'[init] {num_ranks} ranks, device={device}', flush=True)
        print(f'[config] num_tokens={args.num_tokens}, hidden={args.hidden}, '
              f'num_topk={args.num_topk}, num_experts={args.num_experts}, '
              f'repeat={args.repeat}, warmup={args.warmup}', flush=True)
        print(f'[config] dtype=int32 (exact verification, no precision loss)', flush=True)

    buffer = deep_ep.Buffer(group, int(1e9), 0, low_latency_mode=False)

    # Build routing tables (shared across all iterations)
    topk_idx, topk_weights, is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert = \
        build_deterministic_routing(args.num_tokens, args.num_topk, args.num_experts, num_ranks, device)

    all_num_tokens_per_rank = [torch.zeros_like(num_tokens_per_rank) for _ in range(num_ranks)]
    dist.all_gather(all_num_tokens_per_rank, num_tokens_per_rank, group=group)

    all_is_token_in_rank = [torch.zeros_like(is_token_in_rank) for _ in range(num_ranks)]
    dist.all_gather(all_is_token_in_rank, is_token_in_rank, group=group)

    if rank == 0:
        for r in range(num_ranks):
            print(f'[routing] rank {r} sends: {all_num_tokens_per_rank[r].tolist()}', flush=True)

    # Warmup
    if args.warmup > 0:
        if rank == 0:
            print(f'\n[warmup] {args.warmup} iterations...', flush=True)
        for i in range(args.warmup):
            x = build_encoded_data_int32(args.num_tokens, args.hidden, rank, device)
            config = deep_ep.Config(args.num_eus, 256, 384)
            _ = buffer.dispatch(
                x=x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                config=config,
            )
            torch.xpu.synchronize()
        if rank == 0:
            print(f'[warmup] done', flush=True)

    # Profiler run (before test iterations)
    if args.profile and rank == 0:
        print(f'\n[profile] Running profiler...', flush=True)
        x_prof = build_encoded_data_int32(args.num_tokens, args.hidden, rank, device)
        config_prof = deep_ep.Config(args.num_eus, 256, 512)
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            _ = buffer.dispatch(
                x=x_prof,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                config=config_prof,
            )
            torch.xpu.synchronize()
        print(prof.key_averages().table(sort_by='self_xpu_time_total', row_limit=30), flush=True)
        trace_path = f'/tmp/dispatch_trace_rank{rank}.json'
        prof.export_chrome_trace(trace_path)
        print(f'[profile] Trace exported to {trace_path}', flush=True)
    elif args.profile:
        # Non-rank-0 still needs to participate in dispatch
        x_prof = build_encoded_data_int32(args.num_tokens, args.hidden, rank, device)
        config_prof = deep_ep.Config(args.num_eus, 256, 512)
        torch.xpu.synchronize()
        _ = buffer.dispatch(
            x=x_prof,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=config_prof,
        )
        torch.xpu.synchronize()

    if args.profile:
        dist.barrier(group=group)

    # Test iterations
    total_errors = 0
    times_ms = []
    all_passed_local = True

    for iteration in range(args.repeat):
        errs, elapsed_ms = run_dispatch_test(
            args, rank, num_ranks, group, device, buffer, iteration,
            topk_idx, topk_weights, is_token_in_rank,
            num_tokens_per_rank, num_tokens_per_expert,
            all_num_tokens_per_rank, all_is_token_in_rank,
        )
        times_ms.append(elapsed_ms)

        if errs:
            all_passed_local = False
            total_errors += len(errs)
            shown = errs[:args.max_errors]
            for e in shown:
                print(f'[rank {rank}] iter {iteration} ERROR: {e}', flush=True)
            if len(errs) > args.max_errors:
                print(f'[rank {rank}] iter {iteration} ... and {len(errs) - args.max_errors} more errors',
                      flush=True)
        else:
            if args.repeat <= 10 or iteration == 0 or iteration == args.repeat - 1:
                print(f'[rank {rank}] iter {iteration} OK ({elapsed_ms:.3f} ms)', flush=True)

    # Summary
    torch.xpu.synchronize()
    dist.barrier(group=group)

    if times_ms:
        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        print(f'[rank {rank}] {args.repeat} iters: avg={avg_ms:.3f}ms min={min_ms:.3f}ms '
              f'max={max_ms:.3f}ms errors={total_errors}', flush=True)

    passed_tensor = torch.tensor([int(all_passed_local)], dtype=torch.int32, device=device)
    dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
    all_passed = bool(passed_tensor.item())

    if rank == 0:
        if all_passed:
            print(f'\n========== ALL PASSED ==========\n', flush=True)
        else:
            print(f'\n========== FAILED ==========\n', flush=True)

    try:
        dist.destroy_process_group()
    except Exception:
        pass

    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
