import random

import torch

M = 7 # 256
N = 256 # 4096
K = 128 # 7168
E = 17 # 257

torch.manual_seed(42)
random.seed(42)
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


def test_flashinfer_moe():
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


if __name__ == "__main__":
    f_out = test_flashinfer_moe()
    s_out = test_sglang_moe()
    torch.testing.assert_close(f_out, s_out, rtol=1e-2, atol=1e-2)
