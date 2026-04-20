---
name: build-and-env
description: "Set up the build environment and compile DeepEP XPU. Use when: building the project, running setup.py, sourcing environment, activating conda, running tests with mpirun. Trigger phrases: build, compile, env, environment, setup, mpirun, test."
---

# Build & Environment for DeepEP XPU

## Environment Setup

Before any build or test command, source the environment in the **same shell**:

```bash
source ~/zhenyuan/DeepEP/env.sh
```

This sets:
- `USE_XPU=1`, `USE_CUDA=0`
- `XPU_AOT_TARGETS=bmg`, `TORCH_XPU_ARCH_LIST=bmg`
- Intel oneAPI toolchain (icpx compiler via `setvars.sh`)
- Conda environment `lzy`

**CRITICAL**: The environment must be sourced in the same shell as the build/test command. Use `bash -c 'source ~/zhenyuan/DeepEP/env.sh && <command>'` when running from a fresh terminal.

## Build Commands

```bash
cd /home/sdp/zhenyuan/frameworks.ai.pytorch.deepep
bash -c 'source ~/zhenyuan/DeepEP/env.sh && python setup.py build_ext --inplace 2>&1 | tail -60'
```

## Running Tests

All tests require `ZE_AFFINITY_MASK` to be exported **before** the run command. Set it in the shell first, then launch with `mpirun`:

```bash
export ZE_AFFINITY_MASK=0,1,2,3
export RenderCompressedBuffersEnabled=0
export NEOReadDebugKeys=1

bash -c 'source ~/zhenyuan/DeepEP/env.sh && cd /home/sdp/zhenyuan/frameworks.ai.pytorch.deepep && \
  mpirun -np <N> python tests/<test_file>.py <args>'
```

**Always confirm with the user before running**:
- Number of processes (`-np`)
- `ZE_AFFINITY_MASK` value (e.g., `0123` for 4 GPUs)
- Additional environment variables
- Test script arguments
