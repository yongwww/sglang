import random
import torch
import time
import argparse
import os

# Test configuration
M = 256  # 256
N = 1024  # 4096
K = 7168  # 7168
E = 257  # 257

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Generate test data
x_a = torch.rand([M, K], device="cuda", dtype=torch.bfloat16)
w1_q = torch.rand([E, N, K], device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
w2_q = torch.rand([E, K, N // 2], device="cuda", dtype=torch.float32).to(
    torch.float8_e4m3fn
)
w1_scale = torch.rand([E, N // 128, K // 128], device="cuda", dtype=torch.float32)
w2_scale = torch.rand([E, K // 128, N // 256], device="cuda", dtype=torch.float32)
topk_weights = torch.rand([M, 9], device="cuda", dtype=torch.float32)
topk_ids = torch.tensor(
    [random.sample(range(E), 9) for _ in range(M)], device="cuda", dtype=torch.int32
).contiguous()
out_dtype = torch.bfloat16
out = torch.empty([M, K], dtype=out_dtype, device=x_a.device)


def warmup_gpu():
    """Warmup GPU to ensure stable performance measurements."""
    print("Warming up GPU...")
    dummy = torch.randn(1000, 1000, device="cuda")
    for _ in range(10):
        torch.mm(dummy, dummy)
    torch.cuda.synchronize()
    print("GPU warmup completed.")


def benchmark_function(func, name, num_warmup=5, num_runs=100):
    """
    Benchmark a function with proper warmup and timing.

    Args:
        func: Function to benchmark
        name: Name for logging
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Tuple of (result, avg_time_ms)
    """
    print(f"\n=== Benchmarking {name} ===")

    # Warmup runs
    print(f"Performing {num_warmup} warmup runs...")
    result = None
    for i in range(num_warmup):
        result = func()
        torch.cuda.synchronize()

    # Benchmark runs
    print(f"Performing {num_runs} benchmark runs...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_runs):
        result = func()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    print(f"{name} average time: {avg_time_ms:.3f} ms")

    return result, avg_time_ms


def test_flashinfer_moe():
    """Test FlashInfer MoE implementation."""
    from sglang.srt.layers.moe.fused_moe_flashinfer import fused_experts_flashinfer
    flashinfer_out = fused_experts_flashinfer(
        x_a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )
    return flashinfer_out


def test_sglang_moe():
    """Test SGLang MoE implementation."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl
    sgl_out = fused_experts_impl(
        x_a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    return sgl_out


def profile_with_nsys(func, name, num_runs=10):
    """
    Profile a function with nsys markers for detailed analysis.

    nsys profile -o moe_profile -t cuda,nvtx --trace-fork-before-exec=true --cuda-memory-usage=true

    Args:
        func: Function to profile
        name: Name for nsys markers
        num_runs: Number of profiling runs
    """
    print(f"\n=== Profiling {name} with nsys ===")

    # Import nsys if available
    try:
        import nvtx
        nsys_available = True
        print("NVTX available - using detailed markers")
    except ImportError:
        nsys_available = False
        print("NVTX not available - using basic CUDA events")

    # Warmup
    print("Warmup for profiling...")
    for _ in range(3):
        func()
    torch.cuda.synchronize()

    # Profile with markers
    print(f"Profiling {num_runs} runs...")
    torch.cuda.synchronize()

    if nsys_available:
        # Use NVTX markers for detailed profiling
        nvtx.range_push(f"{name}_total")

        for i in range(num_runs):
            nvtx.range_push(f"{name}_run_{i}")
            result = func()
            nvtx.range_pop()

        nvtx.range_pop()
    else:
        # Fallback to basic profiling
        for i in range(num_runs):
            result = func()

    torch.cuda.synchronize()
    print(f"Profiling completed for {name}")
    return result


def run_correctness_test():
    """Run correctness test to ensure both implementations match."""
    print("\n=== Running Correctness Test ===")

    try:
        f_out = test_flashinfer_moe()
        s_out = test_sglang_moe()

        torch.testing.assert_close(f_out, s_out, rtol=1e-2, atol=1e-2)
        print("✅ Correctness test PASSED - outputs match within tolerance")
        return True

    except Exception as e:
        print(f"❌ Correctness test FAILED: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmark comparison."""
    print("\n=== Running Performance Benchmark ===")

    # Benchmark both implementations
    f_out, f_time = benchmark_function(test_flashinfer_moe, "FlashInfer MoE")
    s_out, s_time = benchmark_function(test_sglang_moe, "SGLang MoE")

    # Compare performance
    speedup = s_time / f_time if f_time > 0 else float('inf')
    print(f"\n=== Performance Summary ===")
    print(f"FlashInfer MoE: {f_time:.3f} ms")
    print(f"SGLang MoE:     {s_time:.3f} ms")
    print(f"Speedup:        {speedup:.2f}x ({'FlashInfer faster' if speedup > 1 else 'SGLang faster'})")

    return f_time, s_time


def profile_flashinfer_only():
    """Profile only FlashInfer MoE implementation."""
    print("=== Profiling FlashInfer MoE Only ===")
    warmup_gpu()
    profile_with_nsys(test_flashinfer_moe, "FlashInfer_MoE", num_runs=50)
    print("FlashInfer profiling completed")


def profile_sglang_only():
    """Profile only SGLang MoE implementation."""
    print("=== Profiling SGLang MoE Only ===")
    warmup_gpu()
    profile_with_nsys(test_sglang_moe, "SGLang_MoE", num_runs=50)
    print("SGLang profiling completed")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="MoE Performance Comparison with Nsys Profiling")
    parser.add_argument("--profile-flashinfer", action="store_true",
                       help="Profile only FlashInfer MoE (for separate .nsys-rep)")
    parser.add_argument("--profile-sglang", action="store_true",
                       help="Profile only SGLang MoE (for separate .nsys-rep)")
    parser.add_argument("--skip-correctness", action="store_true",
                       help="Skip correctness test")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip performance benchmark")
    parser.add_argument("--warmup-runs", type=int, default=5,
                       help="Number of warmup runs (default: 5)")
    parser.add_argument("--benchmark-runs", type=int, default=100,
                       help="Number of benchmark runs (default: 100)")
    parser.add_argument("--profile-runs", type=int, default=10,
                       help="Number of profiling runs (default: 10)")

    args = parser.parse_args()

    # Handle separate profiling modes
    if args.profile_flashinfer:
        profile_flashinfer_only()
        return

    if args.profile_sglang:
        profile_sglang_only()
        return

    print("=== MoE Performance Analysis ===")
    print(f"Configuration: M={M}, N={N}, K={K}, E={E}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Warmup GPU
    warmup_gpu()

    # Run correctness test
    if not args.skip_correctness:
        if not run_correctness_test():
            print("❌ Exiting due to correctness test failure")
            return

    # Run performance benchmark
    if not args.skip_benchmark:
        run_performance_benchmark()


if __name__ == "__main__":
    main()
