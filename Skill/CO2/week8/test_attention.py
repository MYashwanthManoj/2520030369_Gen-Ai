"""
Unit tests for Scaled Dot-Product Attention and Multi-Head Attention (unittest compatible).
"""

import unittest
import torch
import torch.nn as nn
from attention import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_causal_mask,
)


class TestAttentionMechanisms(unittest.TestCase):

    def test_scaled_dot_product_attention_shapes(self):
        batch = 2
        heads = 4
        seq_len = 6
        d_k = 16

        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)

        out, weights = scaled_dot_product_attention(q, k, v)

        self.assertEqual(out.shape, (batch, heads, seq_len, d_k))
        self.assertEqual(weights.shape, (batch, heads, seq_len, seq_len))

        # Probabilities must sum to 1 along the key dimension
        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_causal_masking(self):
        seq_len = 5
        d_k = 8
        q = torch.randn(1, 1, seq_len, d_k)
        k = torch.randn(1, 1, seq_len, d_k)
        v = torch.randn(1, 1, seq_len, d_k)

        mask = create_causal_mask(seq_len)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Check that upper triangular elements are strictly 0.0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertEqual(weights[0, 0, i, j].item(), 0.0)

    def test_multi_head_attention_shapes(self):
        batch = 4
        seq_len = 8
        d_model = 64
        num_heads = 8

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(batch, seq_len, d_model)

        out, weights = mha(x, x, x, return_attention=True)

        self.assertEqual(out.shape, (batch, seq_len, d_model))
        self.assertEqual(weights.shape, (batch, num_heads, seq_len, seq_len))

    def test_cross_attention_shapes(self):
        batch = 2
        seq_len_q = 7
        seq_len_kv = 10
        d_model = 32
        num_heads = 4

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        q = torch.randn(batch, seq_len_q, d_model)
        k = torch.randn(batch, seq_len_kv, d_model)
        v = torch.randn(batch, seq_len_kv, d_model)

        out, weights = mha(q, k, v, return_attention=True)

        self.assertEqual(out.shape, (batch, seq_len_q, d_model))
        self.assertEqual(weights.shape, (batch, num_heads, seq_len_q, seq_len_kv))

    def test_numerical_equivalence_with_pytorch_native(self):
        torch.manual_seed(123)
        d_model = 32
        num_heads = 4
        batch = 2
        seq_len = 4

        custom_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=False)
        custom_mha.eval()

        torch_mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True
        )
        torch_mha.eval()

        with torch.no_grad():
            torch_mha.in_proj_weight.copy_(
                torch.cat([custom_mha.w_q.weight, custom_mha.w_k.weight, custom_mha.w_v.weight], dim=0)
            )
            torch_mha.out_proj.weight.copy_(custom_mha.w_o.weight)

        x = torch.randn(batch, seq_len, d_model)

        with torch.no_grad():
            custom_out, _ = custom_mha(x, x, x)
            torch_out, _ = torch_mha(x, x, x)

        diff = (custom_out - torch_out).abs().max().item()
        self.assertLess(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
