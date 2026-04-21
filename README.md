# DeepEP XPU — MoE Communication Primitives for Intel GPU

DeepEP XPU is an Intel GPU (SYCL) port of the [DeepEP](https://github.com/deepseek-ai/DeepEP) library, providing high-throughput **Mixture of Experts (MoE) dispatch and combine** communication primitives via intranode IPC on Intel discrete GPUs (Battlemage).

## Requirements

| Category | Requirement |
|----------|-------------|
| **GPU** | Intel Arc (Battlemage / BMG) |
| **Compiler** | Intel oneAPI DPC++ (`icpx`) with SYCL support |
| **Python** | 3.10+ |
| **PyTorch** | PyTorch with XPU backend (`torch.xpu`) |
| **Optional** | mpi4py (only needed when using `mpirun` launcher) |

## Environment Setup

```bash
# Option A: use the project env script
source env.sh

# Option B: manual setup
source /path/to/oneapi/setvars.sh
conda activate <your_env>
export XPU_AOT_TARGETS=bmg TORCH_XPU_ARCH_LIST=bmg
```

## Build

```bash
cd /path/to/frameworks.ai.pytorch.deepep
git checkout deepep_xpu
python setup.py build_ext --inplace
```

The build produces `deep_ep_cpp.cpython-*.so` in the project root. The `XPU_AOT_TARGETS` environment variable controls the target GPU architecture (default: `bmg`).

## Python API

The main entry point is `deep_ep_xpu.Buffer`:

```python
import deep_ep_xpu as deep_ep

# Core classes
deep_ep.Buffer    # Communication buffer (dispatch / combine)
deep_ep.Config    # Performance tuning config (num_eus, chunk sizes)
deep_ep.topk_idx_t  # torch.int64 — dtype for expert indices
```

### Key Methods

| Method | Description |
|--------|-------------|
| `Buffer(group, num_ipc_bytes, ...)` | Initialize IPC communication buffer |
| `buffer.get_dispatch_layout(topk_idx, num_experts)` | Compute token-to-rank/expert distribution |
| `buffer.dispatch(x, ..., config)` | All-to-all dispatch tokens to target ranks |
| `buffer.combine(x, handle, ..., config)` | Reduce (sum) tokens back to source ranks |
| `Buffer.get_dispatch_config(num_ranks)` | Get recommended dispatch config |
| `Buffer.get_combine_config(num_ranks)` | Get recommended combine config |

## End-to-End Example

A complete **layout → dispatch → combine** pipeline:

```python
import torch
import torch.distributed as dist
import deep_ep_xpu as deep_ep

# --- Initialization (after dist.init_process_group) ---
group = dist.new_group(list(range(dist.get_world_size())))
buffer = deep_ep.Buffer(group, num_ipc_bytes=int(1e9), num_rdma_bytes=0)

rank = dist.get_rank()
num_ranks = dist.get_world_size()
device = f'xpu:{rank}'

num_tokens = 128
hidden = 5120
num_experts = 128
num_topk = 2

# --- Step 1: Build routing (from your MoE gate) ---
# topk_idx: [num_tokens, num_topk], expert index selected by each token
# topk_weights: [num_tokens, num_topk], gating weights
topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk),
                         dtype=deep_ep.topk_idx_t, device=device)
topk_weights = torch.randn(num_tokens, num_topk, dtype=torch.float32, device=device).softmax(dim=-1)

# --- Step 2: Compute dispatch layout ---
num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
    buffer.get_dispatch_layout(topk_idx, num_experts)

# --- Step 3: Dispatch (send tokens to target ranks) ---
x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)

recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, _ = \
    buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )

# --- Step 4: Expert computation (your MoE FFN) ---
# recv_x: [num_recv_tokens, hidden] — tokens assigned to local experts
# Apply your expert FFN here, e.g.:
#   expert_output = expert_ffn(recv_x, recv_topk_idx, num_recv_per_expert)
expert_output = recv_x  # placeholder: identity

# --- Step 5: Combine (reduce results back to source ranks) ---
combined_x, combined_topk_weights, _ = buffer.combine(
    x=expert_output,
    handle=handle,
    topk_weights=recv_topk_weights,
)

# combined_x: [num_tokens, hidden] — each token's expert outputs summed
# combined_topk_weights: [num_tokens, num_topk] — weights for downstream use
```

## Running Tests

### Launcher Configuration

Tests support two launchers, controlled by the `DEEPEP_LAUNCHER` environment variable:

| Value | Launcher |
|-------|----------|
| `mpi` (default) | `mpirun` |
| `torchrun` | `torchrun` |

### Environment Variables

```bash
export ZE_AFFINITY_MASK=0,1,2,3       # GPU device visibility
export RenderCompressedBuffersEnabled=0
export NEOReadDebugKeys=1
export DEEPEP_LAUNCHER=torchrun        # or omit for mpirun
```

### Test Scripts

| Test | Script | Description |
|------|--------|-------------|
| Barrier | `test_xpu_barrier_stress.py` | IPC barrier latency (perf / profile) |
| Layout | `test_xpu_layout.py` | `get_dispatch_layout` correctness |
| Notify Dispatch | `test_xpu_notify_dispatch_stress.py` | notify_dispatch metadata correctness (verify / perf / profile) |
| Dispatch | `test_xpu_intranode_dispatch_stress.py` | Full dispatch correctness (int32 exact verify) |
| Combine | `test_xpu_combine_stress.py` | Dispatch + combine pipeline (verify / perf, int32 / bfloat16) |

### Example Commands

```bash
# --- torchrun ---
export DEEPEP_LAUNCHER=torchrun

# Barrier perf
torchrun --nproc_per_node=4 tests/test_xpu_barrier_stress.py --inner-repeat 100 --output perf

# Notify dispatch verify
torchrun --nproc_per_node=4 tests/test_xpu_notify_dispatch_stress.py --mode verify

# Dispatch verify (int32)
torchrun --nproc_per_node=4 tests/test_xpu_intranode_dispatch_stress.py --num-tokens 16

# Combine verify (int32)
torchrun --nproc_per_node=4 tests/test_xpu_combine_stress.py --dtype int32 --mode verify --num-tokens 16

# Combine perf (bfloat16)
torchrun --nproc_per_node=4 tests/test_xpu_combine_stress.py --dtype bfloat16 --mode perf --num-tokens 4096 --repeat 20

# --- mpirun (no DEEPEP_LAUNCHER needed) ---
mpirun -np 4 python tests/test_xpu_combine_stress.py --dtype int32 --mode verify
```

## Performance Benchmark

Measured on **Intel Arc Pro B60 × 8**, intranode IPC path, `hidden=5120, topk=2, num_experts=128, warmup=3, repeat=5`.

```bash
# Example: 8 cards, 4096 tokens
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
export RenderCompressedBuffersEnabled=0
export NEOReadDebugKeys=1
bash -c 'source /path/to/setvars.sh 2>/dev/null && eval "$(conda shell.bash hook)" && conda activate <env> && \
  cd /path/to/frameworks.ai.pytorch.deepep && \
  DEEPEP_LAUNCHER=torchrun torchrun --nproc_per_node=8 tests/test_xpu_combine_stress.py \
  --dtype bfloat16 --mode perf --num-tokens 4096 --num-experts 128 --repeat 5'
```

| GPUs | num_tokens | dispatch avg | combine avg |
|------|-----------|-------------|-------------|
| 4 | 16 | ~0.81 ms | ~0.87 ms |
| 4 | 4096 | ~54 ms | ~156 ms |
| 8 | 16 | ~0.96 ms | ~0.97 ms |
| 8 | 4096 | ~80 ms | ~199 ms |

> **Note**: The 16-token case is latency-bound (~160 KB of data, dominated by kernel launch overhead). The 4096-token case enters the bandwidth-bound regime; latency increases slightly with more GPUs due to sparser token distribution across experts per rank.

## Supported dtypes

| dtype | Verify (correctness) | Perf (benchmark) |
|-------|---------------------|------------------|
| `int32` | Yes (exact equality) | Yes |
| `bfloat16` | No (precision limitation) | Yes |

## Project Structure

```
csrc/
├── deep_ep.cpp              # Pybind11 bindings + host-side kernel launchers
├── deep_ep.hpp              # C++ declarations
├── event.hpp                # Event handle wrapper
└── sycl/
    ├── api.hpp              # Intranode API declarations
    ├── buffer.hpp           # C++ Buffer runtime class
    ├── configs.h            # Compile-time constants (WARP_SIZE, MAX_RANKS, etc.)
    ├── intranode.cpp         # SYCL kernels: dispatch, combine, barrier
    ├── layout.cpp           # SYCL kernels: dispatch layout computation
    ├── layout.hpp           # Layout kernel declarations
    └── utils.hpp            # Device utilities (atomic ops, barriers, memory access)

deep_ep_xpu/
├── __init__.py              # Exports: Buffer, Config, topk_idx_t
├── buffer.py                # Python Buffer wrapper (dispatch, combine, layout)
└── utils.py                 # EventOverlap utility

tests/
├── test_xpu_barrier_stress.py
├── test_xpu_combine_stress.py
├── test_xpu_intranode_dispatch_stress.py
├── test_xpu_layout.py
└── test_xpu_notify_dispatch_stress.py
```
