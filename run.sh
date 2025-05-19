#!/bin/bash

: ${MODEL=${1:-"deepseek-ai/DeepSeek-R1"}} # HF model id
: ${TP=${2:-8}}                            # tensor-parallel
: ${EP=${4:-1}}                            # expert-parallel - (MOE)
: ${PP=${3:-1}}                            # pipeline-parallel - (AsyncLLMEngine)-WIP
: ${DP=${5:-1}}                            # data-parallel - (MLA)-WIP
: ${DUMMY=${6:--1}}                        # number dummy layers - default -1 to disable dummy weight
: ${IN=${7:-1024}}                         # input prompt length
: ${OUT=${8:-4}}                           # output generate length
: ${BS=${9:-1}}                            # batch size
: ${BEAM=${10:-1}}                         # beam width
: ${PROFILE=${11:-0}}                      # enable profile
: ${XPU_CCL_BACKEND=${12:-"xccl"}}         # ccl, xccl
: ${DISTRIBUTED_BACKEND=${13:-"mp"}}       # uni(tp=1), mp(single node & tp>1), ray(multi nodes)
: ${BENCHMARK=${14:-0}}                    # benchmark LATENCY | THROUGHPUT | SERVER | CLIENT | ACCURACY

MAX_MODEL_LEN=2048
export VLLM_USE_V1=1
export VLLM_MLA_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1
if [[ ${PROFILE} -eq 1 ]]; then
  export VLLM_TORCH_PROFILER_DIR=${PWD}/profile
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NUM_DUMMY_LAYERS=${DUMMY}
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE # TP8+COMPOSITE | TP16+FLAT
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export XPU_CCL_BACKEND=${XPU_CCL_BACKEND}

export OPT_W8A8_BLOCK_FP8_MATMUL=1

if [[ ${TP} -le 1 ]]; then
  DISTRIBUTED_BACKEND=uni
else
  DISTRIBUTED_BACKEND=mp
fi
if [[ ${PP} -gt 1 ]]; then
  DISTRIBUTED_BACKEND=ray
  export VLLM_USE_V1=0
fi

case ${BENCHMARK} in
"LATENCY")
  CMD="python benchmarks/benchmark_latency.py \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len ${MAX_MODEL_LEN} \
    --distributed-executor-backend ${DISTRIBUTED_BACKEND} \
    --tensor-parallel-size ${TP} \
    --input-len ${IN} \
    --output-len ${OUT} \
    --batch-size ${BS} \
    --n ${BEAM} \
    --num-iters-warmup 3 \
    --num-iters 5"

  [ ${BEAM} -ne 1 ] && CMD+=" --use-beam-search"
  [ ${EP} -ge 1 ] && CMD+=" --enable-expert-parallel"
  [ ${PROFILE} -eq 1 ] && CMD+=" --profile --profile-result-dir ${VLLM_TORCH_PROFILER_DIR}"
  ;;
"THROUGHPUT")
  CMD="python benchmarks/benchmark_throughput.py \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len ${MAX_MODEL_LEN} \
    --distributed-executor-backend ${DISTRIBUTED_BACKEND} \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP} \
    --input-len ${IN} \
    --output-len ${OUT} \
    --backend vllm \
    --n ${BEAM} \
    --num-prompts 200"

  [ ${PP} -gt 1 ] CMD+=" --async-engine"
  [ ${EP} -ge 1 ] && CMD+=" --enable-expert-parallel"
  [ ${PROFILE} -eq 1 ] && CMD+=" --profile"
  ;;
"SERVER")
  CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len ${MAX_MODEL_LEN} \
    --distributed-executor-backend ${DISTRIBUTED_BACKEND} \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP} \
    --dtype float16 \
    --device xpu \
    --port 8000 \
    --block-size 32 \
    --gpu-memory-util 0.95 \
    --max_num_batched_tokens 8192 \
    --max-num-seqs 29 \
    --no-enable-prefix-caching"

  [ ${EP} -ge 1 ] && CMD+=" --enable-expert-parallel"
  ;;
"CLIENT")
  CMD="python benchmarks/benchmark_serving.py \
    --model ${MODEL} \
    --backend vllm \
    --host localhost \
    --port 8000 \
    --dataset-name random \
    --num-prompts 3 \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency 16 \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --random-range-ratio 0.8"

  [ ${BEAM} -ne 1 ] && CMD+=" --use-beam-search"
  [ ${PROFILE} -eq 1 ] && CMD+=" --profile"
  ;;
"ACCURACY")
  CMD="lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},distributed_executor_backend=${DISTRIBUTED_BACKEND},tensor_parallel_size=${TP},trust_remote_code=true,enforce_eager=true,max_model_len=4096" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --limit 32 \
    --batch_size ${BS}"
  ;;
*)
  CMD="python examples/offline_inference/cli.py \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len ${MAX_MODEL_LEN} \
    --distributed-executor-backend ${DISTRIBUTED_BACKEND} \
    --tensor-parallel-size ${TP} \
    --input-len ${IN} \
    --output-len ${OUT}"

  [ ${EP} -ge 1 ] && CMD+=" --enable-expert-parallel"
  ;;
esac

echo CMD=${CMD}

date
eval ${CMD}
date
