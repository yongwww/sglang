import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sgl_kernel import silu_and_mul

import torch

import flashinfer
from flashinfer.gemm import group_gemm_fp8_nt_groupwise


def fused_experts_flashinfer(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.
    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - out_dtype (torch.Tensor): The output tensor type.
    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """    
    print(
        f"Input shapes of fused_experts_flashinfer:\n"
        f"  a: {a.shape} (dtype: {a.dtype})\n"
        f"  w1_q: {w1_q.shape} (dtype: {w1_q.dtype})\n"
        f"  w2_q: {w2_q.shape} (dtype: {w2_q.dtype})\n"
        f"  w1_scale: {w1_scale.shape} (dtype: {w1_scale.dtype})\n"
        f"  w2_scale: {w2_scale.shape} (dtype: {w2_scale.dtype})\n"
        f"  topk_weights: {topk_weights.shape} (dtype: {topk_weights.dtype})\n"
        f"  topk_ids: {topk_ids.shape} (dtype: {topk_ids.dtype})\n"
    )

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[-1], "Hidden size mismatch"
    assert w1_q.shape[1] == w2_q.shape[2] * 2, "Hidden size mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    # a: torch.Size([256, 7168]),
    # w1_q: torch.Size([257, 4096, 7168]),
    # w2_q: torch.Size([257, 7168, 2048]),
    # w1_scale: torch.Size([257, 32, 56]),
    # w2_scale: torch.Size([257, 56, 16]),
    # topk_weights: torch.Size([256, 9]),
    # topk_ids: torch.Size([256, 9])
    # rep_a_q: torch.Size([2304, 7168]),
    # w1_q: torch.Size([257, 4096, 7168]),
    # topk_ids: torch.Size([256, 9])
    # rep_a1_scales: torch.Size([2304, 56]),
    # w1_scale: torch.Size([257, 32, 56])

    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    M = a.size(0) # 256
    N = w1_q.size(1) # 4096
    K = w2_q.size(1) # 7168

    E = w1_q.size(0) # 257

    top_k = topk_ids.size(1) # 9


    # preparation
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, 1, E
    )

    # Group gemm 1
    a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)
    # a_q: torch.Size([256, 7168]), dtype: torch.float8_e4m3fn, device: cuda:0
    # a1_scale: torch.Size([256, 56]), dtype: torch.float32
    device = a_q.device
    A1 = torch.zeros((M * top_k, K), dtype=a_q.dtype, device=device)  # Keep as fp8
    A1[::] = a_q[sorted_token_ids // top_k]  # Use quantized a_q
    # Blocksize = 1
    output, count = expert_ids.unique_consecutive(return_counts=True)
    C1 = torch.zeros((M * top_k, N), dtype=out_dtype, device=device)
    m_indptr = torch.zeros((E + 1,), dtype=torch.int32, device=device)
    m_indptr[output + 1] = count.int()
    m_indptr = m_indptr.cumsum(dim=0, dtype=m_indptr.dtype)

    group_gemm_fp8_nt_groupwise(a=A1, b=w1_q, a_scale=a1_scale, b_scale=w1_scale, m_indptr=m_indptr, out=C1)
    # C1: torch.Size([2304, 4096]), dtype: torch.bfloat16

    # silu and mul
    intermediate = torch.empty((M * top_k, N * 2), device=device, dtype=out_dtype)
    silu_and_mul(intermediate, C1)

    # Group gemm 2
    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(
        intermediate, 128)

    C2 = torch.zeros((M * top_k, K), dtype=out_dtype, device=device)

    group_gemm_fp8_nt_groupwise(a=intemediate_q, b=w2_q, a_scale=a2_scale, b_scale=w2_scale, m_indptr=m_indptr, out=C2)

    # Create a new tensor in bfloat16 for the final result
    res = torch.zeros((M * top_k, K), dtype=out_dtype, device=device)
    res[sorted_token_ids] = C2[::]
    return res.view((M, top_k, K)).sum(dim=1)
