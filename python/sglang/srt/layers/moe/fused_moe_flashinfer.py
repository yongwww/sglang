import functools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sgl_kernel import silu_and_mul

import torch

# import flashinfer
# from flashinfer.gemm import group_gemm_fp8_nt_groupwise

import torch
import triton
import triton.language as tl


@triton.jit
def compute_one_hot(topk_ids, one_hot, M: tl.constexpr, top_k: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(0, top_k):
        e = tl.load(topk_ids + pid * top_k + i)
        tl.store(one_hot + e * M + pid, 1)


@triton.jit
def compute_padding_mapping(
    m_indptr,
    padded_m_indptr,
    m_rank,
    padded_m_rank,
):
    pid = tl.program_id(0)
    m_start = tl.load(m_indptr + pid)
    m_end = tl.load(m_indptr + pid + 1)
    padded_m_start = tl.load(padded_m_indptr + pid)
    for i in range(m_end - m_start):
        tl.store(m_rank + m_start + i, m_start + i)
        tl.store(padded_m_rank + m_start + i, padded_m_start + i)


@triton.jit
def compute_ranking(
    topk_ids,
    one_hot_cumsum,
    m_indptr,
    token_ids_ranking,
    M: tl.constexpr,
    top_k: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in range(0, top_k):
        e = tl.load(topk_ids + pid * top_k + i)
        offset = tl.load(one_hot_cumsum + e * M + pid)
        m_start = tl.load(m_indptr + e)
        tl.store(token_ids_ranking + m_start + offset - 1, pid * top_k + i)


def fake_group_gemm(a, b, a_scale, b_scale, m_indptr, out):
    cum_m, k = a.shape
    e, n, _ = b.shape
    assert b.shape[2] == k
    assert a_scale.shape == (k // 128, cum_m)
    assert b_scale.shape == (e, k // 128, n // 128)
    assert out.shape == (cum_m, n)
    assert m_indptr.shape[0] == e + 1
    # assert m_indptr[-1] == cum_m


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
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
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
    device = a.device
    M = a.size(0)  # 256
    N = w1_q.size(1)  # 4096
    K = w2_q.size(1)  # 7168
    E = w1_q.size(0)  # 257

    top_k = topk_ids.size(1)  # 9

    one_hot = torch.zeros((E, M), dtype=torch.int32, device=device)
    compute_one_hot[(M,)](topk_ids, one_hot, M, top_k)
    one_hot_cumsum = one_hot.cumsum(dim=1, dtype=torch.int32)
    expert_activated_count = one_hot_cumsum[::, -1]
    m_len = torch.zeros((E + 1,), dtype=torch.int32, device=device)
    m_len[1:] = expert_activated_count
    m_indptr = m_len.cumsum(dim=0, dtype=torch.int32)
    assert m_indptr[-1] == top_k * M
    padded_m_indptr = (m_len + 3 - (m_len + 3) % 4).cumsum(dim=0, dtype=torch.int32)
    m_rank = torch.zeros((m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device)
    padded_m_rank = torch.zeros(
        (m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device
    )
    compute_padding_mapping[(E,)](m_indptr, padded_m_indptr, m_rank, padded_m_rank)
    token_ids_ranking = torch.zeros((m_indptr[-1],), dtype=torch.int32, device=device)
    compute_ranking[(M,)](
        topk_ids, one_hot_cumsum, m_indptr, token_ids_ranking, M, top_k
    )

    # Group gemm 1
    # a_q: torch.Size([256, 7168]), dtype: torch.float8_e4m3fn, device: cuda:0
    # a1_scale: torch.Size([256, 56]), dtype: torch.float32
    device = a.device
    a1 = torch.zeros(
        (padded_m_indptr[-1], K), dtype=a.dtype, device=device
    )  # Keep as fp8
    a1[padded_m_indptr] = a[token_ids_ranking[m_rank] // top_k]
    # Blocksize = 1
    c1 = torch.zeros((padded_m_indptr[-1], N), dtype=out_dtype, device=device)
    a1_q, a1_scale = sglang_per_token_group_quant_fp8(a1, 128, column_major_scales=True)
    a1_scale = a1_scale.permute(1, 0)
    w1_scale = w1_scale.permute(0, 2, 1).contiguous()
    print(a1_q.shape, a1_scale.shape)
    print(w1_q.shape, w1_scale.shape)
    fake_group_gemm(
        a=a1_q, b=w1_q, a_scale=a1_scale, b_scale=w1_scale, m_indptr=m_indptr, out=c1
    )
    # group_gemm_fp8_nt_groupwise(
    #     a=a1_q, b=w1_q, a_scale=a1_scale, b_scale=w1_scale, m_indptr=m_indptr, out=C1
    # )

    # C1: torch.Size([2304, 4096]), dtype: torch.bfloat16

    # silu and mul
    intermediate = silu_and_mul(c1)

    # Group gemm 2
    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(
        intermediate, 128, column_major_scales=True
    )
    a2_scale = a2_scale.permute(1, 0)
    w2_scale = w2_scale.permute(0, 2, 1).contiguous()

    c2 = torch.zeros((padded_m_indptr[-1], K), dtype=out_dtype, device=device)
    fake_group_gemm(
        a=intemediate_q,
        b=w2_q,
        a_scale=a2_scale,
        b_scale=w2_scale,
        m_indptr=m_indptr,
        out=c2,
    )
    # group_gemm_fp8_nt_groupwise(
    #     a=intemediate_q,
    #     b=w2_q,
    #     a_scale=a2_scale,
    #     b_scale=w2_scale,
    #     m_indptr=m_indptr,
    #     out=C2,
    # )

    res = torch.zeros((m_indptr[-1], K), dtype=c2.dtype, device=c2.device)
    res[token_ids_ranking[m_rank]] = c2[padded_m_rank]
    return res.view((M, top_k, K)).sum(dim=1)
