# LFM3 NVFP4 with FP8 KV on RTX 5090

This branch adds native `Lfm3ForCausalLM` execution and width-7 short
convolutions to vLLM 0.32. It incorporates the model implementation from
[vLLM PR #64959](https://github.com/vllm-project/vllm/pull/64959) and the
convolution extension from [PR #64647](https://github.com/vllm-project/vllm/pull/64647),
with the original authorship retained.

The serving example uses the original ModelOpt NVFP4 checkpoint. Feed-forward
projections use native CUTLASS W4A4 kernels; attention projections,
convolutions, embeddings and norms remain BF16. Attention KV storage is FP8
E4M3. The checkpoint does not contain calibrated KV scales, so the existing
vLLM default of 1.0 applies. Context length is 32,768 tokens per request.

Install this checkout using the development workflow in
[the contributing guide](../../../docs/contributing/README.md). Python and
Triton changes can reuse the vLLM 0.32.0 precompiled CUDA artifacts.

```bash
source .venv/bin/activate
bash examples/online_serving/lfm3_nvfp4_fp8/serve.sh
```

To use an existing local checkpoint, set `LFM3_MODEL` to its directory. Set
`LFM3_PORT` to change the localhost port. Further arguments are passed to vLLM.

The example captures full decode CUDA graphs while leaving prefill eager.
This avoids a full Inductor compilation during startup. Use `--enforce-eager`
for a simpler diagnostic run. Neither mode implies a particular startup time;
measure process launch through a completed first response on the target machine.
The server runs in the foreground and vLLM owns its worker-process lifecycle.

See [local validation results](VALIDATION.md) for precision inspection, tests,
and measured process startup. The under-three-second startup target is unmet.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm3-1.7b-nvfp4","messages":[{"role":"user","content":"What is the capital of France? Reply with the city name only."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```
