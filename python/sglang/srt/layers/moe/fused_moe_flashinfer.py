import flashinfer
import torch
import triton
import triton.language as tl
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from sgl_kernel import silu_and_mul

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


@triton.jit
def compute_one_hot(topk_ids, one_hot, M: tl.constexpr, top_k: tl.constexpr):
    """
    Compute one-hot encoding for top-k expert selections.
    """
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
    """
    Compute mapping between original and padded indices for memory alignment.
    """
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
    """
    Compute ranking of tokens for grouped operations.
    """
    pid = tl.program_id(0)
    for i in range(0, top_k):
        e = tl.load(topk_ids + pid * top_k + i)
        offset = tl.load(one_hot_cumsum + e * M + pid)
        m_start = tl.load(m_indptr + e)
        tl.store(token_ids_ranking + m_start + offset - 1, pid * top_k + i)


def fused_experts_flashinfer(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a8w8-quantized Mixture of Experts (MoE) layer using FlashInfer backend. 

    The key innovation is using grouped GEMM to efficiently batch computations
    for tokens assigned to the same expert, avoiding the overhead of sparse
    operations while maintaining the sparsity benefits of MoE.

    Args:
        a (torch.Tensor): Input activations with shape [M, K] where:
            - M is the number of tokens (sequence length * batch size)
            - K is the hidden dimension
        w1_q (torch.Tensor): First expert weights (FP8 quantized) with shape [num_experts, K, 2N]:
            - num_experts is the total number of experts
            - K is the input hidden dimension
            - 2N is twice the intermediate dimension (for SwiGLU gating)
            - Weights are stored in transposed format for efficient GEMM
        w2_q (torch.Tensor): Second expert weights (FP8 quantized) with shape [num_experts, N, K]:
            - N is the intermediate dimension
            - Weights are stored in transposed format
        w1_scale (torch.Tensor): Dequantization scales for w1_q with shape:
            - [num_experts] for per-expert scaling, or
            - [num_experts, 2N] for per-expert per-channel scaling
        w2_scale (torch.Tensor): Dequantization scales for w2_q with shape:
            - [num_experts] for per-expert scaling, or
            - [num_experts, K] for per-expert per-channel scaling
        topk_weights (torch.Tensor): Routing weights for selected experts [M, top_k]:
            - Contains the softmax-normalized weights for each token-expert pair
        topk_ids (torch.Tensor): Selected expert indices [M, top_k]:
            - Contains the expert IDs chosen by the gating network
            - Must have dtype torch.int32

    Returns:
        torch.Tensor: MoE layer output with shape [M, K], same dtype as input `a`
    """
    assert topk_weights.shape == topk_ids.shape, f"topk shape mismatch: weights {topk_weights.shape} vs ids {topk_ids.shape}"
    assert w1_q.dtype == torch.float8_e4m3fn, f"w1_q must be FP8, got {w1_q.dtype}"
    assert w2_q.dtype == torch.float8_e4m3fn, f"w2_q must be FP8, got {w2_q.dtype}"
    assert a.shape[1] == w1_q.shape[-1], f"Input-weight dimension mismatch: {a.shape[1]} vs {w1_q.shape[-1]}"
    assert w1_q.shape[1] == w2_q.shape[2] * 2, f"Weight dimension mismatch: w1 {w1_q.shape[1]} vs w2*2 {w2_q.shape[2] * 2}"
    assert w1_q.shape[0] == w2_q.shape[0], f"Expert count mismatch: {w1_q.shape[0]} vs {w2_q.shape[0]}"
    assert w1_q.shape[0] == w1_scale.shape[0], f"w1 scale expert count mismatch: {w1_q.shape[0]} vs {w1_scale.shape[0]}"
    assert w1_q.shape[0] == w2_scale.shape[0], f"w2 scale expert count mismatch: {w1_q.shape[0]} vs {w2_scale.shape[0]}"
    assert a.dtype in [torch.half, torch.bfloat16], f"Invalid input dtype: {a.dtype}"
    assert w1_q.is_contiguous(), "Expert weights w1_q must be contiguous"
    assert w2_q.is_contiguous(), "Expert weights w2_q must be contiguous"

    # Extract tensor properties
    out_dtype = a.dtype
    device = a.device
    M = a.size(0)  # Number of tokens
    N = w1_q.size(1)  # Intermediate dimension * 2 (for SwiGLU)
    K = w2_q.size(1)  # Hidden dimension
    E = w1_q.size(0)  # Number of experts
    top_k = topk_ids.size(1)  # Number of experts per token

    # Build one-hot matrix indicating which experts are selected for each token
    one_hot = torch.zeros((E, M), dtype=torch.int32, device=device)
    compute_one_hot[(M,)](topk_ids, one_hot, M, top_k)
    one_hot_cumsum = one_hot.cumsum(dim=1, dtype=torch.int32)
    expert_activated_count = one_hot_cumsum[::, -1]  # Total tokens per expert

    # Create indptr arrays for grouped GEMM
    m_len = torch.zeros((E + 1,), dtype=torch.int32, device=device)
    m_len[1:] = expert_activated_count
    m_indptr = m_len.cumsum(dim=0, dtype=torch.int32)

    # pad to multiple of 4
    padded_m_indptr = (m_len + 3 - (m_len + 3) % 4).cumsum(dim=0, dtype=torch.int32)

    # Compute mappings between original and padded indices
    m_rank = torch.zeros((M * top_k,), dtype=m_indptr.dtype, device=m_indptr.device)
    padded_m_rank = torch.zeros((M * top_k,), dtype=m_indptr.dtype, device=m_indptr.device)
    compute_padding_mapping[(E,)](m_indptr, padded_m_indptr, m_rank, padded_m_rank)

    # Determine the order of token processing for efficient batching
    token_ids_ranking = torch.zeros((M * top_k,), dtype=torch.int32, device=device)
    compute_ranking[(M,)](topk_ids, one_hot_cumsum, m_indptr, token_ids_ranking, M, top_k)

    # Allocate workspace tensors with padding
    max_m = top_k * M + 3 * E + 3 - (top_k * M + 3 * E + 3) % 4
    cache_mk = torch.zeros((max_m, K), dtype=a.dtype, device=device)
    cache_mn = torch.zeros((max_m, N), dtype=out_dtype, device=device)

    # Arrange input tokens according to expert grouping
    a1 = cache_mk
    a1[padded_m_rank] = a[token_ids_ranking[m_rank] // top_k]

    # Quantize input activations to FP8 with groupwise scaling
    a1_q, a1_scale = sglang_per_token_group_quant_fp8(a1, 128, column_major_scales=True)
    a1_scale = a1_scale.permute(1, 0)  # Transpose for GEMM compatibility
    w1_scale = w1_scale.permute(0, 2, 1).contiguous()

    # Execute first grouped GEMM: a1_q @ w1_q -> c1
    c1 = cache_mn
    group_gemm_fp8_nt_groupwise(
        a=a1_q,
        b=w1_q,
        a_scale=a1_scale,
        b_scale=w1_scale,
        m_indptr=padded_m_indptr,
        out=c1,
    )

    intermediate = silu_and_mul(c1)

    # Quantize intermediate activations
    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(
        intermediate, 128, column_major_scales=True
    )
    a2_scale = a2_scale.permute(1, 0)  # Transpose for GEMM compatibility
    w2_scale = w2_scale.permute(0, 2, 1).contiguous()  # Prepare weight scales

    # Execute second grouped GEMM: intermediate_q @ w2_q -> c2
    c2 = cache_mk
    group_gemm_fp8_nt_groupwise(
        a=intemediate_q,
        b=w2_q,
        a_scale=a2_scale,
        b_scale=w2_scale,
        m_indptr=padded_m_indptr,
        out=c2,
    )

    # Rearrange expert outputs back to original token order
    res = torch.zeros((M * top_k, K), dtype=c2.dtype, device=c2.device)
    res[token_ids_ranking[m_rank]] = c2[padded_m_rank]

    # Reshape to [M, top_k, K] and apply routing weights
    res = res.view(M, top_k, K)
    weighted = res * topk_weights.to(res.dtype).unsqueeze(-1)  # [M, top_k, K]
    return weighted.sum(dim=1)  # [M, K]
