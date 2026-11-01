# Local validation

Measured on 2026-11-01 with an RTX 5090 (32 GiB), driver 615.71.09,
Python 3.11, PyTorch 2.13.0+cu130 and vLLM 0.32.0 precompiled CUDA
artifacts. Model and Triton code came from this branch. No C++ changes or
subagents were used. The main installation and Pi integration were unchanged.

Checkpoint: `LiquidAI/LFM3-1.7B-NVFP4` at revision
`5f8f594ffc22400b02a11e00e2d9393a9371c161`. The safetensors SHA256 is
`386de2dd415909869ad7d72b734783b687f0d41f7dbbc519d9ae4257883a4ba2`.
Tests used seed 42, synthetic inputs, and no training or external compute.

## Precision and correctness

A trusted local `LLM.apply_model` inspection verified all 48 fused FFN
projections use packed NVFP4 weights and `CutlassNvFp4LinearKernel`.
All 11 attention layers reported `fp8_e4m3` KV caches, physically stored
as byte tensors. Remaining checkpoint layers retain BF16 precision.
No BF16 FFN fallback or conversion to another weight format was used.
KV scales were the default 1.0; calibrated scales are absent from this checkpoint.

- 312 causal-convolution tests passed, including width 7 and existing widths.
- Two model registry/import tests passed.
- Seven ModelOpt mixed-NVFP4 configuration tests passed.
- Real-checkpoint arithmetic, factual and copying smoke checks passed.
- A 32,512-input-token middle-key retrieval passed in both offline inference
  and the HTTP API. The configured context limit was 32,768 tokens.
- Four concurrent short chat requests, streaming output, an automatic weather
  tool call, and its tool-result round trip passed through the HTTP API.

These are smoke and kernel checks, not an accuracy benchmark or a demonstration
of parity with BF16 or SGLang. Arithmetic wording differed from the earlier
SGLang smoke output. Throughput parity and a concurrency sweep at 32K have not
been measured for this fork.

## Startup

Each timing starts with a fully stopped server process and ends after a
completed short factual answer through the OpenAI-compatible API. Filesystem
and compiled-kernel caches were populated. No model, CUDA context or serving
process was retained between runs. The configuration is `serve.sh`, using
full decode CUDA graphs and eager prefill.

| Run | HTTP ready (s) | First completed answer (s) |
| --- | ---: | ---: |
| Cached 1 | 19.911 | 19.944 |
| Cached 2 | 20.369 | 20.398 |
| Cached 3 | 19.202 | 19.234 |

The median first-answer time was **19.944 seconds**. An earlier graph setup
run took 21.212 seconds. A separate eager offline run reached model-ready in
11.509 seconds; that excludes the serving API and is not comparable to the
end-to-end timing. The **under-three-second goal is unmet**. Native model
support by itself does not remove vLLM's process, import and engine setup costs.

Peak total system RAM during the HTTP checks was 19.79 GB. A 28.5 GB guard
monitored the tests. All test servers were stopped after completion.

Initial diagnostic attempts exposed a disabled trusted-local callback, a
custom GCC sysroot missing the CUDA driver link path, and a test-client assumption
that `/health` returns JSON. These were corrected in the local test setup;
the final HTTP run and all three repeats completed successfully with exit 0.
No insecure callback serialization was enabled on the HTTP server.

## Test commands

Run these from the checkout with the documented development environment:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest \
  tests/kernels/mamba/test_causal_conv1d.py -q
.venv/bin/python -m pytest \
  'tests/models/test_registry.py::test_registry_imports[Lfm3ForCausalLM]' \
  tests/models/test_registry.py::test_hf_registry_coverage -q
.venv/bin/python -m pytest tests/config/test_modelopt_mixed_nvfp4.py -q
```

The implementation derives from the credited upstream model and kernel PRs
linked in [the serving instructions](README.md). This fork was integrated and
validated with AI assistance; it is not a separate upstream contribution.
