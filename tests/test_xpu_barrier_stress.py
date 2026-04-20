#!/usr/bin/env python
"""
XPU Barrier Stress Test

Modes:
  perf:     C++ inner loop + host timing, reports avg barrier latency
  profile:  Python loop with single barrier + torch.profiler JSON trace

Optional --verify: run write→barrier→verify data correctness check first

Example commands:
    # mpirun (default, DEEPEP_LAUNCHER=mpi)
    mpirun -np 2 python tests/test_xpu_barrier_stress.py --use-mpi --inner-repeat 1000 --output perf
    mpirun -np 2 python tests/test_xpu_barrier_stress.py --use-mpi --inner-repeat 1000 --output perf --verify
    mpirun -np 2 python tests/test_xpu_barrier_stress.py --use-mpi --inner-repeat 10 --output profile --trace-dir ./traces
    # torchrun (set DEEPEP_LAUNCHER=torchrun)
    DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=4 tests/test_xpu_barrier_stress.py --inner-repeat 100 --output perf
"""

import argparse
import os
import time
import torch
import torch.distributed as dist


LAUNCHER = os.environ.get('DEEPEP_LAUNCHER', 'mpi').lower()

if LAUNCHER == 'mpi':
    try:
        from mpi4py import MPI
        HAS_MPI = True
    except ImportError:
        HAS_MPI = False
        MPI = None
else:
    HAS_MPI = False
    MPI = None


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
    except Exception as e:
        print(f"[Warning] xccl failed, trying gloo: {e}")
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{port}',
            world_size=world_size, rank=rank)

    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device, comm


def init_dist_torchrun():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.xpu.set_device(local_rank)
    device = f'xpu:{local_rank}'

    dist.init_process_group(backend='xccl')

    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group, device


def test_loop(args):
    import sys
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.absolute()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import deep_ep_xpu as deep_ep

    inner_repeat = args.inner_repeat
    output = args.output

    if output == 'profile':
        assert args.outer_repeat == 1, \
            f"--outer-repeat must be 1 in profile mode (got {args.outer_repeat})"

    if LAUNCHER == 'torchrun':
        rank, num_ranks, group, device = init_dist_torchrun()
        mpi_comm = None
    else:
        rank, num_ranks, group, device, mpi_comm = init_dist_mpi(args.port)

    if rank == 0:
        print(f"\n{'#'*60}")
        print(f"# XPU Barrier Stress Test")
        print(f"# Ranks: {num_ranks}, Device: {device}")
        print(f"# inner_repeat={inner_repeat}, outer_repeat={args.outer_repeat}")
        print(f"# warmup={args.warmup}")
        print(f"# output={output}, verify={args.verify}")
        print(f"{'#'*60}\n")

    if mpi_comm is not None:
        buffer = deep_ep.Buffer(
            group, int(1e8), 0,
            low_latency_mode=False,
            comm=mpi_comm)
    else:
        buffer = deep_ep.Buffer(
            group, int(1e8), 0,
            low_latency_mode=False)
    if rank == 0:
        print(f"[Info] Buffer created (launcher={LAUNCHER})")

    dist.barrier(group=group)

    def cpu_barrier():
        if mpi_comm is not None:
            mpi_comm.Barrier()
        else:
            dist.barrier(group=group)

    try:
        # === Phase 1: verify (optional) ===
        if args.verify:
            if rank == 0:
                print(f"\n[Verify] data correctness check "
                      f"(inner={inner_repeat}, data_size={args.data_size})...")
            cpu_barrier()
            torch.xpu.synchronize()
            cpu_barrier()

            total_errors = 0
            for outer in range(args.outer_repeat):
                errors = buffer.runtime.test_barrier_stress(
                    inner_repeat, outer * inner_repeat, args.data_size)
                total_errors += errors
                if errors > 0:
                    print(f"[Rank {rank}] outer {outer+1}: {errors} errors!", flush=True)

            cpu_barrier()
            if rank == 0:
                if total_errors > 0:
                    raise RuntimeError(f"Verify FAILED: {total_errors} errors")
                print(f"[Verify] PASSED: {args.outer_repeat * inner_repeat} barriers, 0 errors\n")

        # === Phase 2: warmup ===
        if args.warmup > 0:
            if rank == 0:
                print(f"[Warmup] {args.warmup} barriers...")
            cpu_barrier()
            torch.xpu.synchronize()
            cpu_barrier()
            buffer.runtime.test_barrier_perf(args.warmup)
            cpu_barrier()
            if rank == 0:
                print(f"[Warmup] done")

        # === Phase 3: run ===
        cpu_barrier()
        torch.xpu.synchronize()
        cpu_barrier()

        total_barriers = inner_repeat * args.outer_repeat

        if output == 'perf':
            t0 = time.time()
            for outer in range(args.outer_repeat):
                buffer.runtime.test_barrier_perf(inner_repeat)
            elapsed = time.time() - t0

            cpu_barrier()
            if rank == 0:
                avg_us = elapsed * 1e6 / total_barriers
                print(f"\n{'#'*60}")
                print(f"# PASSED: {total_barriers} barriers in "
                      f"{elapsed*1000:.1f} ms ({avg_us:.1f} us/barrier)")
                print(f"{'#'*60}\n")

        elif output == 'profile':
            from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

            trace_dir = args.trace_dir
            os.makedirs(trace_dir, exist_ok=True)

            activities = [ProfilerActivity.CPU]
            if hasattr(ProfilerActivity, 'XPU'):
                activities.append(ProfilerActivity.XPU)

            with profile(
                activities=activities,
                on_trace_ready=tensorboard_trace_handler(trace_dir),
                record_shapes=False, with_stack=False,
            ) as prof:
                for i in range(inner_repeat):
                    buffer.runtime.test_barrier()
                    prof.step()

            cpu_barrier()
            if rank == 0:
                json_files = [f for f in os.listdir(trace_dir) if f.endswith('.json')]
                print(f"\n{'#'*60}")
                print(f"# DONE: {inner_repeat} barriers profiled, "
                      f"{len(json_files)} trace files in {trace_dir}/")
                print(f"{'#'*60}\n")

    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        dist.barrier(group=group)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='XPU Barrier Stress Test')
    parser.add_argument('--inner-repeat', type=int, default=100,
                        help='Barrier count per C++ call (default: 100)')
    parser.add_argument('--outer-repeat', type=int, default=1,
                        help='Python outer loop count (default: 1)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup barrier count before timing (default: 10)')
    parser.add_argument('--output', type=str, default='perf',
                        choices=['perf', 'profile'],
                        help='perf: host timing; profile: torch profiler JSON')
    parser.add_argument('--verify', action='store_true',
                        help='Run data correctness verification first')
    parser.add_argument('--data-size', type=int, default=1024,
                        help='Data size in ints for verify (default: 1024)')
    parser.add_argument('--trace-dir', type=str, default='./barrier_traces',
                        help='Profile trace output dir (default: ./barrier_traces)')
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--use-mpi', action='store_true')
    args = parser.parse_args()

    if LAUNCHER == 'mpi' and not HAS_MPI:
        print("Error: mpi4py not installed (set DEEPEP_LAUNCHER=torchrun to use torchrun)")
        return
    test_loop(args)


if __name__ == '__main__':
    main()
