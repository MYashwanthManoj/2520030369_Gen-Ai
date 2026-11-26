# Module II - Week 2: Attention Mechanisms from Scratch

This directory contains pure PyTorch implementations of **Scaled Dot-Product Attention** and **Multi-Head Attention (MHA)** from scratch, following the foundational Transformer architecture introduced in *"Attention Is All You Need"* (Vaswani et al., 2017).

---

## 1. Mathematical Formulation

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

- **$Q \in \mathbb{R}^{B \times H \times N_q \times d_k}$**: Queries
- **$K \in \mathbb{R}^{B \times H \times N_k \times d_k}$**: Keys
- **$V \in \mathbb{R}^{B \times H \times N_k \times d_v}$**: Values
- **$\sqrt{d_k}$**: Scaling factor to prevent dot products from growing excessively large, which pushes the softmax function into regions with vanishing gradients.
- **$M$**: Attention mask (e.g., Causal/Autoregressive upper-triangular mask with $-\infty$ or padding mask).

### Multi-Head Attention (MHA)
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

---

## 2. Directory Structure

- `attention.py`: Core clean implementations:
  - `scaled_dot_product_attention(...)` functional implementation
  - `ScaledDotProductAttention(nn.Module)` wrapper
  - `MultiHeadAttention(nn.Module)` full module from scratch
  - `create_causal_mask(...)` utility for autoregressive decoding
- `demo.py`: Interactive script walking through each component, causal masking, cross-attention, and numerical verification against PyTorch's native `nn.MultiheadAttention`.
- `test_attention.py`: Unit test suite verifying tensor dimensions, probability normalization, causal masking invariants, and numerical equivalence.
- `requirements.txt`: Package dependencies.

---

## 3. Running the Code

### Run Interactive Demo
```bash
python demo.py
```

### Run Unit Tests
```bash
python test_attention.py
```
