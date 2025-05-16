import torch

M = 256
N = 4096
K = 7168
E = 257

torch.manual_seed(42)
x_a = torch.rand([M, K], device='cuda').to(torch.bfloat16)
w1_q = torch.rand([E, N, K], device="cuda").to(torch.float8_e4m3fn)
w2_q = torch.rand([E, K, N // 2], device="cuda").to(torch.float8_e4m3fn)
w1_scale = torch.rand([E, 32, 56], device="cuda").to(torch.float32)
w2_scale = torch.rand([E, 56, 16], device="cuda").to(torch.float32)
topk_weights =  torch.rand([M, 9], device="cuda").to(torch.float32) 
topk_ids =  torch.rand([M, 9], device="cuda").to(torch.int32)

out_dtype = torch.bfloat16
out = torch.empty([834, 4096], dtype=out_dtype, device=x_a.device)

def test_flashinfer_moe():
    from sglang.srt.layers.moe.fused_moe_flashinfer import fused_experts_flashinfer
    flashinfer_out =  fused_experts_flashinfer(
        x_a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )
    print(f"flashinfer_out: {flashinfer_out}\nshape: {flashinfer_out.shape}, dtype: {flashinfer_out.dtype}")


def test_sglang_moe():
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl
    sgl_out = fused_experts_impl(
        x_a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        inplace=False,
        activation = "silu",
        apply_router_weight_on_input = False,
        use_fp8_w8a8 = True,
        w1_scale = w1_scale,
        w2_scale = w2_scale,
        block_shape = [128, 128],
    )
    print(f"sgl_out: {sgl_out}\nshape: {sgl_out.shape}, dtype: {sgl_out.dtype}")

    """
    Input tensor shapes:
    hidden_states: torch.Size([256, 7168]) (dtype: torch.bfloat16)
    w1: torch.Size([257, 4096, 7168]) (dtype: torch.float8_e4m3fn)
    w2: torch.Size([257, 7168, 2048]) (dtype: torch.float8_e4m3fn)
    topk_weights: torch.Size([256, 9]) (dtype: torch.float32)
    topk_ids: torch.Size([256, 9]) (dtype: torch.int32)
    w1_scale: torch.Size([257, 32, 56]) (dtype: torch.float32)
    w2_scale: torch.Size([257, 56, 16]) (dtype: torch.float32)

    inplace: True
    activation: silu
    apply_router_weight_on_input: False
    use_fp8_w8a8: True
    use_int8_w8a8: False
    use_int8_w8a16: False
    use_int4_w4a16: False
    per_channel_quant: False
    block_shape: [128, 128]
    no_combine: False
    """

if __name__ == "__main__":
    test_flashinfer_moe()
    test_sglang_moe()
