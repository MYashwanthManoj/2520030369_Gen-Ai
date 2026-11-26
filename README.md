# Generative AI Coursework & Labs

This repository contains the laboratory assignments and practical implementations for the Generative AI course.

## Repository Structure

```text
├── Skill/
│   ├── week1-llm-call/              # Week 1: First LLM API Call (OpenAI / Gemini)
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── .env.example
│   ├── Week-2/                      # Week 2: Provider-Agnostic LLM Client (Hosted & Ollama)
│   │   ├── llm_client.py
│   │   ├── main.py
│   │   ├── test_client.py
│   │   ├── requirements.txt
│   │   └── .env.example
│   ├── Week3/                       # Week 3: LLM Monitoring with MongoDB
│   │   ├── llm_client.py
│   │   ├── usage_logger.py
│   │   ├── main.py
│   │   ├── test_client.py
│   │   ├── requirements.txt
│   │   └── .env.example
│   ├── Week4/                       # Week 4: Subword Tokenizer Comparison (English, Telugu, Hindi)
│   │   ├── tokenizer_comparison.py
│   │   └── requirements.txt
│   └── CO2/                         # Course Outcome 2 Labs
│       ├── week7/                   # Week 7 (CO2 Week 1): Next-Token Probabilities & Logits (GPT-2)
│       │   ├── next_token_prob.py
│       │   ├── requirements.txt
│       │   └── gpt2_next_token_results.csv
│       └── week8/                   # Week 8 (CO2 Week 2): Attention Mechanisms from Scratch (Pure PyTorch)
│           ├── attention.py
│           ├── demo.py
│           ├── interactive_attention.py
│           ├── test_attention.py
│           ├── requirements.txt
│           └── README.md
```

## Lab Overviews

### Week 1: First LLM API Call
- Connects a Python application to Google Gemini / OpenAI via API.
- Secure environment management with `.env`.

### Week 2: Provider-Agnostic LLM Client
- Builds a unified client interface (`LLMClient`, `LLMProvider`) over hosted API (Gemini) and local LLMs (Ollama with `llama3.2`).
- Unit tests included with `test_client.py`.

### Week 3: LLM Observability & Monitoring with MongoDB
- Logs token usage, latency (ms), and estimated API cost for every call into MongoDB (`llm_monitoring.api_usage`).
- Implements exponential backoff retries and timeout protection.

### Week 4: Tokenizer Comparison (BPE vs. WordPiece vs. SentencePiece)
- Compares subword tokenization across English, Telugu, and Hindi using:
  - Byte-Pair Encoding (`gpt2`)
  - WordPiece (`bert-base-multilingual-cased`)
  - SentencePiece (`xlm-roberta-base`)

### CO2 Week 7: Next-Token Probabilities, Logits & Decoding Analysis (GPT-2)
- Loads `openai-community/gpt2` to analyze subword tokenization and compute raw logit vectors across all 50,257 vocabulary tokens.
- Explores temperature scaling ($T = 0.2, 1.0, 2.0$) and softmax to control deterministic vs. creative decoding distributions.
- Extracts top-$K$ candidate tokens, visualizes probability bars, and exports all experiment runs to CSV.

### CO2 Week 8: Attention Mechanisms from Scratch (Pure PyTorch)
- Implements core Transformer building blocks from scratch following *"Attention Is All You Need"* (Vaswani et al., 2017).
- **Scaled Dot-Product Attention:** $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$.
- **Causal Masking:** Autoregressive upper-triangular masking for generative decoding.
- **Multi-Head Attention (MHA):** Multi-head linear projections, head splitting, scaled dot-product attention, and concatenation projection.
- **Cross-Attention & Equivalence:** Verified mathematically against PyTorch native `torch.nn.MultiheadAttention`.
