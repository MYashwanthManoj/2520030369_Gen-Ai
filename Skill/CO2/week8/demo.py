"""
Interactive Demo & Verification for Scaled Dot-Product & Multi-Head Attention.
Demonstrates:
1. Mathematical intuition behind Scaled Dot-Product Attention step-by-step
2. Self-attention with causal (autoregressive) masking
3. Multi-Head Attention forward pass and shape verification
4. Cross-attention (Encoder-Decoder) example
5. Numerical comparison against PyTorch's native nn.MultiheadAttention
"""

import torch
import torch.nn as nn
from attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
)


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)


def demo_scaled_dot_product():
    print_header("1. SCALED DOT-PRODUCT ATTENTION (Step-by-Step Toy Example)")

    torch.manual_seed(42)
    seq_len = 3
    d_k = 4

    # Toy words representation
    tokens = ["The", "cat", "sat"]
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    print(f"Tokens: {tokens}")
    print(f"Input Shape: (batch_size=1, seq_len={seq_len}, d_k={d_k})")

    out, weights = scaled_dot_product_attention(q, k, v)

    print("\nAttention Matrix (Weights):")
    for i, token in enumerate(tokens):
        row_str = " | ".join(f"{w.item():.4f}" for w in weights[0, i])
        print(f"  {token:5s} -> [{row_str}]")

    print("\nRow sums (verify sum to 1.0):", weights.sum(dim=-1).squeeze().tolist())
    print(f"Output Shape: {tuple(out.shape)}")


def demo_causal_masking():
    print_header("2. CAUSAL (AUTOREGRESSIVE) MASKING")

    seq_len = 4
    d_k = 4
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)

    mask = create_causal_mask(seq_len)
    print("Causal Mask (True = masked / cannot attend to future tokens):")
    print(mask.int().numpy())

    out, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    print("\nMasked Attention Matrix (Upper triangle is 0.0):")
    print(weights.squeeze(0).detach().numpy().round(4))


def demo_multi_head_attention():
    print_header("3. MULTI-HEAD ATTENTION (MHA) FORWARD PASS")

    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    mha.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    causal_mask = create_causal_mask(seq_len)

    out, attn_weights = mha(x, x, x, mask=causal_mask, return_attention=True)

    print(f"Input Tensor Shape  : {tuple(x.shape)} (batch, seq, d_model)")
    print(f"Number of Heads     : {num_heads}")
    print(f"Dimension per Head  : {d_model // num_heads}")
    print(f"Output Tensor Shape : {tuple(out.shape)}")
    print(f"Attention Map Shape : {tuple(attn_weights.shape)} (batch, heads, seq, seq)")


def demo_cross_attention():
    print_header("4. CROSS-ATTENTION (ENCODER-DECODER ATTENTION)")

    batch_size = 2
    seq_len_q = 4  # Decoder target tokens
    seq_len_kv = 6  # Encoder source tokens
    d_model = 32
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_kv, d_model)
    v = torch.randn(batch_size, seq_len_kv, d_model)

    out, attn_weights = mha(query=q, key=k, value=v, return_attention=True)

    print(f"Query (Decoder) Shape : {tuple(q.shape)}")
    print(f"Key/Val (Encoder) Shape: {tuple(k.shape)}")
    print(f"Cross-Attention Out   : {tuple(out.shape)} (matches Query sequence length)")
    print(f"Attention Weights     : {tuple(attn_weights.shape)} (seq_len_q x seq_len_kv)")


def demo_benchmark_with_pytorch_native():
    print_header("5. EQUIVALENCE CHECK WITH TORCH.NN.MULTIHEADATTENTION")

    torch.manual_seed(0)
    d_model = 32
    num_heads = 4
    batch_size = 2
    seq_len = 3

    # Custom implementation
    custom_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=False)
    custom_mha.eval()

    # PyTorch native
    torch_mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True
    )
    torch_mha.eval()

    # Copy weights from custom to PyTorch native to compare exact math
    with torch.no_grad():
        # In PyTorch native, in_proj_weight is [3*d_model, d_model] = [W_q; W_k; W_v]
        torch_mha.in_proj_weight.copy_(
            torch.cat([custom_mha.w_q.weight, custom_mha.w_k.weight, custom_mha.w_v.weight], dim=0)
        )
        torch_mha.out_proj.weight.copy_(custom_mha.w_o.weight)

    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        custom_out, custom_weights = custom_mha(x, x, x, return_attention=True)
        torch_out, torch_weights = torch_mha(x, x, x, need_weights=True, average_attn_weights=False)

    max_diff = (custom_out - torch_out).abs().max().item()
    print(f"Max absolute difference between Custom & PyTorch Native output: {max_diff:.8e}")
    assert max_diff < 1e-5, f"Discrepancy too large: {max_diff}"
    print("MATCH CONFIRMED! Implementation is mathematically identical to PyTorch native.")


if __name__ == "__main__":
    demo_scaled_dot_product()
    demo_causal_masking()
    demo_multi_head_attention()
    demo_cross_attention()
    demo_benchmark_with_pytorch_native()
