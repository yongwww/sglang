import triton
import triton.language as tl


@triton.jit
def _copy_contiguous_bytes_kernel(
    src_ptr,
    dst_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _copy_strided_bytes_kernel(
    src_ptr,
    dst_ptr,
    src_stride0: tl.constexpr,
    src_stride1: tl.constexpr,
    dst_stride0: tl.constexpr,
    dst_stride1: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    PAGE_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < PAGE_COUNT * PAGE_BYTES
    page = linear // PAGE_BYTES
    byte = linear - page * PAGE_BYTES
    values = tl.load(
        src_ptr + page * src_stride0 + byte * src_stride1,
        mask=mask,
        other=0,
    )
    tl.store(
        dst_ptr + page * dst_stride0 + byte * dst_stride1,
        values,
        mask=mask,
    )


@triton.jit
def _fused_index_k_kernel(
    k_ptr,
    loc_ptr,
    weight_ptr,
    bias_ptr,
    cos_sin_ptr,
    positions_ptr,
    output_ptr,
    token_count,
    eps,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    loc_stride0: tl.constexpr,
    weight_stride0: tl.constexpr,
    bias_stride0: tl.constexpr,
    cos_stride0: tl.constexpr,
    cos_stride1: tl.constexpr,
    positions_stride0: tl.constexpr,
    output_stride0: tl.constexpr,
    output_stride1: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    row_mask = rows < token_count

    x = tl.load(
        k_ptr + rows[:, None] * k_stride0 + cols[None, :] * k_stride1,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / HEAD_DIM)
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) * (1.0 / HEAD_DIM)
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols * weight_stride0)[None, :]
    bias = tl.load(bias_ptr + cols * bias_stride0)[None, :]
    values = centered * rstd[:, None] * weight + bias

    pairs = tl.reshape(values, BLOCK_M, 64, 2)
    real, imag = tl.split(pairs)
    pair_cols = tl.arange(0, 64)[None, :]
    rotated = pair_cols < 32
    position = tl.load(
        positions_ptr + rows * positions_stride0,
        mask=row_mask,
        other=0,
    )[:, None]
    rope_mask = row_mask[:, None] & rotated
    cos = tl.load(
        cos_sin_ptr + position * cos_stride0 + pair_cols * cos_stride1,
        mask=rope_mask,
        other=1.0,
    )
    sin = tl.load(
        cos_sin_ptr
        + position * cos_stride0
        + (32 + pair_cols) * cos_stride1,
        mask=rope_mask,
        other=0.0,
    )
    real_out = tl.where(rotated, real * cos - imag * sin, real)
    imag_out = tl.where(rotated, real * sin + imag * cos, imag)
    values = tl.reshape(tl.join(real_out, imag_out), BLOCK_M, HEAD_DIM)

    rounded = values.to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    amax = tl.max(tl.abs(rounded), axis=1)
    scale = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
    quantized = tl.minimum(
        tl.maximum(rounded / scale[:, None], -448.0), 448.0
    )
    fp8_bits = quantized.to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    ).to(tl.uint8, bitcast=True)

    location = tl.load(
        loc_ptr + rows * loc_stride0,
        mask=row_mask,
        other=0,
    )
    page = location // PAGE_SIZE
    offset = location - page * PAGE_SIZE
    key_offsets = offset[:, None] * HEAD_DIM + cols[None, :]
    key_ptrs = (
        output_ptr
        + page[:, None] * output_stride0
        + key_offsets * output_stride1
    )
    tl.store(key_ptrs, fp8_bits, mask=row_mask[:, None])

    scale_bits = scale.to(tl.uint32, bitcast=True)
    byte_cols = tl.arange(0, 4)[None, :]
    scale_bytes = (scale_bits[:, None] >> (byte_cols * 8)).to(tl.uint8)
    scale_offsets = HEAD_DIM * PAGE_SIZE + offset[:, None] * 4 + byte_cols
    scale_ptrs = (
        output_ptr
        + page[:, None] * output_stride0
        + scale_offsets * output_stride1
    )
    tl.store(scale_ptrs, scale_bytes, mask=row_mask[:, None])


def run(
    k_input,
    cache,
    out_cache_loc,
    weight,
    bias,
    eps,
    cos_sin_cache,
    positions,
    page_size,
    updated_cache,
):
    cache_elements = cache.numel()
    copy_block = 65536
    if cache.is_contiguous() and updated_cache.is_contiguous():
        _copy_contiguous_bytes_kernel[(triton.cdiv(cache_elements, copy_block),)](
            cache,
            updated_cache,
            n_elements=cache_elements,
            BLOCK=copy_block,
            num_warps=8,
            num_stages=1,
        )

    else:
        _copy_strided_bytes_kernel[(triton.cdiv(cache_elements, copy_block),)](
            cache,
            updated_cache,
            src_stride0=cache.stride(0),
            src_stride1=cache.stride(1),
            dst_stride0=updated_cache.stride(0),
            dst_stride1=updated_cache.stride(1),
            PAGE_BYTES=cache.shape[1],
            PAGE_COUNT=cache.shape[0],
            BLOCK=copy_block,
            num_warps=8,
            num_stages=1,
        )

    token_count = k_input.shape[0]
    block_m = 8
    _fused_index_k_kernel[(triton.cdiv(token_count, block_m),)](
        k_input,
        out_cache_loc,
        weight,
        bias,
        cos_sin_cache,
        positions,
        updated_cache,
        token_count,
        eps,
        k_stride0=k_input.stride(0),
        k_stride1=k_input.stride(1),
        loc_stride0=out_cache_loc.stride(0),
        weight_stride0=weight.stride(0),
        bias_stride0=bias.stride(0),
        cos_stride0=cos_sin_cache.stride(0),
        cos_stride1=cos_sin_cache.stride(1),
        positions_stride0=positions.stride(0),
        output_stride0=updated_cache.stride(0),
        output_stride1=updated_cache.stride(1),
        PAGE_SIZE=page_size,
        HEAD_DIM=128,
        BLOCK_M=block_m,
        num_warps=8,
        num_stages=1,
    )
