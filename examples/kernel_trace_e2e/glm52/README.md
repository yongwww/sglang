# GLM-5.2 Kernel Factory E2E validation

This integration replaces the stock SGLang DSA cache-update operation only
when `SGLANG_KERNEL_TRACE_DSA_CACHE_UPDATE=1` is set. The stock path is
unchanged otherwise.

- The unmodified `elementwise.py` is byte-identical to the source in serving
  image digest
  `sha256:31ea943e1ba6450918ee1a214634f57e21e3d99da9975a322dcf7fa99f865e76`;
  its SHA-256 is
  `96d5c601474900c36a740c2ceaeb6740c7e3e0fc3475abc5d1be92907bbf42ab`.
- `kernel_trace_dsa_cache_update.py` is byte-identical to the Kernel Factory
  artifact; its SHA-256 is
  `f8ad3bd2032ae12cb5462be17e1c6a33a934e34ae8e240374707f27fb878ad0`.
- The generated candidate was targeted at GB300. Its B300 run is therefore a
  correctness/portability and B300-performance observation, not a GB300 claim.
- The runner verifies the pinned InferenceX revision and recipe hash and does
  not rewrite CUDA-graph, speculative-decoding, cache, or request controls.

Example:

```bash
MODE=candidate PHASE=gsm8k \
INFERENCEX_ROOT=/opt/yongwww/runtime-workspaces/kimi-exact-source-d4f825b1 \
MODEL_PATH=/opt/dlami/nvme/models/GLM-5.2-NVFP4 \
CACHE_DIR=/home/scratch.yowu_coreai/kernel-trace/glm52-production-exact-workspace-20260816/container-home/.cache \
RESULT_DIR=/home/scratch.yowu_coreai/kernel-trace/e2e-gsm8k/glm52-candidate \
bash examples/kernel_trace_e2e/glm52/run_exact_inferencex.sh
```

Use `MODE=stock` for the baseline. After GSM8K passes, set `PHASE=perf`; the
script runs the recipe's full 3,600-second AgentX workload at concurrency 8.
Set `VERIFY_ONLY=1` to check all pinned revisions and hashes without launching.

The recipe intentionally disables simulated EAGLE acceptance for `PHASE=gsm8k`
and uses real target-logit verification. Its performance phase preserves the
recipe's committed golden-acceptance controls for a controlled stock/KF A/B.
