#!/usr/bin/env bash
set -euo pipefail

# Reproduce GLM-5.2 stock/KF comparisons with the pinned InferenceX recipe.

MODE="${MODE:?set MODE=stock or MODE=candidate}"
PHASE="${PHASE:?set PHASE=gsm8k or PHASE=perf}"
INFERENCEX_ROOT="${INFERENCEX_ROOT:?set INFERENCEX_ROOT}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
RESULT_DIR="${RESULT_DIR:?set RESULT_DIR}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR}"

IMAGE="${IMAGE:-localhost:5000/kernel-trace/day0-sglang-serving@sha256:31ea943e1ba6450918ee1a214634f57e21e3d99da9975a322dcf7fa99f865e76}"
RECIPE="benchmarks/single_node/agentic/glm5.2_fp4_b300_sglang_mtp.sh"
EXPECTED_RECIPE_SHA256="050d18ce4a8e986171ec47059f41a51a3e9f36a0bd67bd54d54b19845cbec1f0"
EXPECTED_INFERENCEX_REV="d4f825b1101604b5126b18b3ff024207b27c803e"
EXPECTED_INTEGRATION_SHA256="7d9fb1c1b05fc960e806f4ee7e0d916340d52e8eb9a19c8b2b5f2f973d9f3d35"
EXPECTED_CANDIDATE_SHA256="f8ad3bd2032ae12cb5462be17e1c6a33a934e34ae8e240374707f27fb878ad0a"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

case "$MODE" in stock|candidate) ;; *) echo "invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$PHASE" in gsm8k|perf) ;; *) echo "invalid PHASE: $PHASE" >&2; exit 2 ;; esac

test "$(git -C "$INFERENCEX_ROOT" rev-parse HEAD)" = "$EXPECTED_INFERENCEX_REV"
echo "$EXPECTED_RECIPE_SHA256  $INFERENCEX_ROOT/$RECIPE" | sha256sum --check --status
echo "$EXPECTED_INTEGRATION_SHA256  $REPO_ROOT/python/sglang/jit_kernel/dsv32/elementwise.py" | sha256sum --check --status
echo "$EXPECTED_CANDIDATE_SHA256  $REPO_ROOT/python/sglang/jit_kernel/dsv32/kernel_trace_dsa_cache_update.py" | sha256sum --check --status
mkdir -p "$RESULT_DIR" "$CACHE_DIR"

if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
  echo "Pinned GLM-5.2 E2E inputs verified."
  exit 0
fi

docker_args=(
  run --rm --entrypoint bash --gpus all --ipc=host --network host --shm-size=32g
  -w /results
  -v "$INFERENCEX_ROOT:/workspace:ro"
  -v "$MODEL_PATH:$MODEL_PATH:ro"
  -v "$RESULT_DIR:/results"
  -v "$CACHE_DIR:/root/.cache"
  -e MODEL=nvidia/GLM-5.2-NVFP4
  -e MODEL_PATH="$MODEL_PATH"
  -e TP=8 -e CONC=8 -e EP_SIZE=1 -e DP_ATTENTION=false
  -e KV_OFFLOADING=dram -e KV_OFFLOAD_BACKEND=hicache
  -e 'KV_OFFLOAD_BACKEND_METADATA={"name":"hicache"}'
  -e TOTAL_CPU_DRAM_GB=1889
  -e RESULT_DIR=/results -e AGENTIC_OUTPUT_DIR=/results
  -e RESULT_FILENAME="glm52_b300_${MODE}"
  -e DURATION=3600 -e PORT=8888 -e HOME=/root
)

if [[ "$PHASE" == gsm8k ]]; then
  docker_args+=(
    -e EVAL_ONLY=true
    -e EVAL_RESULT_DIR=/results
    -e EVAL_TASKS_DIR=/workspace/utils/evals/gsm8k.yaml
  )
else
  docker_args+=(-e EVAL_ONLY=false)
fi

if [[ "$MODE" == candidate ]]; then
  docker_args+=(
    -e SGLANG_KERNEL_TRACE_DSA_CACHE_UPDATE=1
    -v "$REPO_ROOT/python/sglang/jit_kernel/dsv32/elementwise.py:/sgl-workspace/sglang/python/sglang/jit_kernel/dsv32/elementwise.py:ro"
    -v "$REPO_ROOT/python/sglang/jit_kernel/dsv32/kernel_trace_dsa_cache_update.py:/sgl-workspace/sglang/python/sglang/jit_kernel/dsv32/kernel_trace_dsa_cache_update.py:ro"
  )
fi

{
  echo "framework_repo=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "inferencex_repo=$EXPECTED_INFERENCEX_REV"
  echo "recipe_sha256=$EXPECTED_RECIPE_SHA256"
  echo "image=$IMAGE"
  echo "mode=$MODE"
  echo "phase=$PHASE"
  sha256sum \
    "$REPO_ROOT/python/sglang/jit_kernel/dsv32/elementwise.py" \
    "$REPO_ROOT/python/sglang/jit_kernel/dsv32/kernel_trace_dsa_cache_update.py"
} > "$RESULT_DIR/e2e-run-metadata.txt"

docker "${docker_args[@]}" "$IMAGE" "/workspace/$RECIPE"
