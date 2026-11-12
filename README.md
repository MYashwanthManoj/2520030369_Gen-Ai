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
│   └── Week4/                       # Week 4: Subword Tokenizer Comparison (English, Telugu, Hindi)
│       ├── tokenizer_comparison.py
│       └── requirements.txt
└── test.txt
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
