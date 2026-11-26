"""
Module II - Week 2: Attention Mechanisms from Scratch in PyTorch
Implementation of:
1. Scaled Dot-Product Attention (with optional causal/padding masking and dropout)
2. Multi-Head Attention (MHA) module as described in "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + mask ) @ V

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask broadcastable to (..., seq_len_q, seq_len_k).
              Positions with True / 1 will be masked out (or boolean tensor where True = keep,
              or additive mask where -inf / -1e9 means masked out).
              Here we support boolean mask where True means masked (ignored),
              or float tensor already containing additive bias (e.g. 0.0 for keep, -inf for mask).
        dropout: Optional nn.Dropout layer applied to attention weights.

    Returns:
        output: Attention output tensor of shape (..., seq_len_q, d_v)
        attn_weights: Normalized attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # 1. Matmul: (..., seq_len_q, d_k) x (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scale by sqrt(d_k) to prevent vanishing gradients for large dimensions
    scores = scores / math.sqrt(d_k)

    # 3. Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            # Mask positions where mask == True (standard PyTorch masked_fill convention)
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            # Additive mask (e.g. 0 for attend, -inf for ignore)
            scores = scores + mask

    # 4. Softmax along last dimension (key sequence length)
    attn_weights = F.softmax(scores, dim=-1)

    # 5. Optional dropout on attention probabilities
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 6. Weighted sum: (..., seq_len_q, seq_len_k) x (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    nn.Module wrapper for Scaled Dot-Product Attention.
    """

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout=self.dropout,
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) module from scratch.

    Projects Q, K, V into `num_heads` distinct subspaces, applies scaled dot-product attention
    in parallel across heads, concatenates results, and projects through an output linear layer.

    References:
        "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            d_model: Total embedding dimension of the model.
            num_heads: Number of parallel attention heads (must divide d_model).
            dropout: Dropout probability applied to attention weights.
            bias: Whether linear projection layers include bias.
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        if self.w_q.bias is not None:
            nn.init.zeros_(self.w_q.bias)
            nn.init.zeros_(self.w_k.bias)
            nn.init.zeros_(self.w_v.bias)
            nn.init.zeros_(self.w_o.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Tensor of shape (batch_size, seq_len_q, d_model)
            key: Tensor of shape (batch_size, seq_len_k, d_model)
            value: Tensor of shape (batch_size, seq_len_k, d_model)
            mask: Optional mask tensor. Can be:
                  - 2D: (seq_len_q, seq_len_k)
                  - 3D: (batch_size, seq_len_q, seq_len_k)
                  - 4D: (batch_size, 1, seq_len_q, seq_len_k)
                  True indicates positions that should be masked (ignored).
            return_attention: Whether to return the attention weights tensor.

        Returns:
            output: Tensor of shape (batch_size, seq_len_q, d_model)
            attn_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k) if requested, else None.
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)

        # 1. Linear projections
        # Shape: (batch_size, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2. Reshape into multi-heads:
        # (batch_size, seq_len, num_heads, d_k) -> transpose -> (batch_size, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Format mask for 4D broadcasting: (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
            if mask.dim() == 2:
                # (seq_len_q, seq_len_k) -> (1, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)

        # 4. Scaled Dot-Product Attention per head
        attn_out, attn_weights = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask,
            dropout=self.attn_dropout if self.training else None,
        )

        # 5. Concatenate heads back:
        # (batch_size, num_heads, seq_len_q, d_k) -> transpose -> (batch_size, seq_len_q, num_heads, d_k)
        # contiguous() -> view(batch_size, seq_len_q, d_model)
        concat_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )

        # 6. Final linear projection W_o
        output = self.w_o(concat_out)

        if return_attention:
            return output, attn_weights
        return output, None


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Creates a square causal (autoregressive) mask where future positions are True (masked out).
    Shape: (seq_len, seq_len)
    """
    # torch.triu with diagonal=1 creates upper triangular matrix above diagonal
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
