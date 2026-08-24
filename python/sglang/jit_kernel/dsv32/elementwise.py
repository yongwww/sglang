"""DSA only. Indexer K kernels (JIT)."""

import logging
import os

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

_CUDA_FILE = "deepseek_v32/indexer_k.cuh"
logger = logging.getLogger(__name__)
_kernel_trace_dsa_cache_update_logged = False


@cache_once
def _jit_k_indexer_norm_rope_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "dpsk_v32_k_indexer_norm_rope",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_k_indexer_norm_rope_store_module(dtype: torch.dtype, page_size: int):
    args = make_cpp_args(dtype, is_arch_support_pdl(), page_size)
    return load_jit(
        f"dpsk_v32_k_indexer_norm_rope_store_p{page_size}",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("forward", f"FusedKIndexerNormRopeStoreKernel<{args}>::forward"),
        ],
    )


def fused_k_indexer_norm_rope(
    k_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """V3.2 indexer K: LayerNorm + RoPE on leading dims -> bf16. CUDA only."""
    # k_input may be a non-contiguous wk slice; output is always contiguous.
    k_out = torch.empty(k_input.shape, dtype=k_input.dtype, device=k_input.device)
    module = _jit_k_indexer_norm_rope_module(k_input.dtype)
    module.forward(
        k_input,
        k_out,
        weight,
        bias,
        cos_sin_cache,
        positions,
        float(eps),
    )
    return k_out


def fused_k_indexer_norm_rope_store(
    k_input: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
) -> None:
    """V3.2 indexer K + fused store: LayerNorm + RoPE on leading dims + fp8
    act-quant + paged index-k cache write, in one launch. CUDA only."""
    if os.environ.get("SGLANG_KERNEL_TRACE_DSA_CACHE_UPDATE") == "1":
        global _kernel_trace_dsa_cache_update_logged
        if (
            k_input.dtype != torch.bfloat16
            or k_input.shape[-1] != 128
            or cache.dtype != torch.uint8
            or cache.ndim != 2
            or cache.shape[1] != 132 * page_size
        ):
            raise RuntimeError(
                "Kernel Trace DSA cache update received an unsupported contract"
            )
        if not _kernel_trace_dsa_cache_update_logged:
            logger.info("Using the Kernel Factory DSA cache-update candidate.")
            _kernel_trace_dsa_cache_update_logged = True
        from sglang.jit_kernel.dsv32.kernel_trace_dsa_cache_update import (
            run as kernel_trace_dsa_cache_update,
        )

        kernel_trace_dsa_cache_update(
            k_input,
            cache,
            out_cache_loc,
            weight,
            bias,
            eps,
            cos_sin_cache,
            positions,
            page_size,
            cache,
        )
        return

    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    module = _jit_k_indexer_norm_rope_store_module(k_input.dtype, page_size)
    module.forward(
        k_input,
        cache,
        out_cache_loc,
        weight,
        bias,
        cos_sin_cache,
        positions,
        float(eps),
    )
