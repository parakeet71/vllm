#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

exec vllm serve "${LFM3_MODEL:-LiquidAI/LFM3-1.7B-NVFP4}" \
  --trust-remote-code \
  --served-model-name lfm3-1.7b-nvfp4 \
  --host 127.0.0.1 --port "${LFM3_PORT:-8000}" \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 \
  --linear-backend cutlass --attention-backend TRITON_ATTN \
  --max-model-len 32768 --max-num-seqs 64 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice --generation-config vllm --seed 42 "$@"
