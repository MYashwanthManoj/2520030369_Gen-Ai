"""
Interactive CLI for Scaled Dot-Product & Multi-Head Attention (Module II - Week 2).
Accepts user text input, custom tensor dimensions, or pre-configured prompt/question benchmarks,
computes query/key/value attention matrices from scratch, and displays
the resulting attention weight tables in the terminal.
"""

import math
import torch
from attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
)

# Pre-set questions / benchmark prompts
PRESET_QUESTIONS = [
    {
        "id": 1,
        "title": "Pronoun Disambiguation / Coreference Resolution",
        "sentence": "The animal didn't cross the street because it was too tired",
        "causal": False,
        "heads": 4,
        "description": "Examines whether the token 'it' attends more strongly to 'animal' or 'street'."
    },
    {
        "id": 2,
        "title": "Transformer Definition (Vaswani et al.)",
        "sentence": "Attention is all you need for sequence modeling",
        "causal": False,
        "heads": 4,
        "description": "Standard self-attention over foundational Transformer terminology."
    },
    {
        "id": 3,
        "title": "Autoregressive Causal Next-Token Prediction",
        "sentence": "The capital of France is Paris",
        "causal": True,
        "heads": 4,
        "description": "Causal autoregressive masked attention where each word only looks backward."
    },
    {
        "id": 4,
        "title": "Long-Distance Syntactic Dependency",
        "sentence": "The keys that I left on the kitchen table were missing",
        "causal": False,
        "heads": 4,
        "description": "Examines long-range agreement between subject ('keys') and verb ('were')."
    },
    {
        "id": 5,
        "title": "Polysemy / Context Sensitivity",
        "sentence": "The bank by the river had a steep muddy slope",
        "causal": False,
        "heads": 4,
        "description": "Examines context disambiguation ('bank' attending to 'river' vs financial institution)."
    }
]


def print_attention_table(weights: torch.Tensor, tokens: list[str], title: str = "Attention Matrix"):
    """Prints a formatted ASCII attention weight matrix."""
    print(f"\n{title}")
    col_w = max(max(len(t) for t in tokens), 8) + 2
    header = f"{'Query \\ Key':<{col_w}}" + "".join(f"{t:>{col_w}}" for t in tokens)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, q_tok in enumerate(tokens):
        row = f"{q_tok:<{col_w}}" + "".join(f"{weights[i, j].item():>{col_w}.4f}" for j in range(len(tokens)))
        print(row)
    print("-" * len(header))


def compute_and_show_attention(tokens: list[str], num_heads: int, use_causal: bool):
    seq_len = len(tokens)
    d_k = 16
    d_model = num_heads * d_k

    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    x = torch.randn(1, seq_len, d_model)
    mask = create_causal_mask(seq_len) if use_causal else None

    with torch.no_grad():
        output, attn_weights = mha(x, x, x, mask=mask, return_attention=True)

    print("\n" + "=" * 76)
    print(f"Tokens ({seq_len})         : {tokens}")
    print(f"Causal Masking       : {'Enabled (Autoregressive)' if use_causal else 'Disabled (Bidirectional)'}")
    print(f"Input Shape          : (batch=1, seq_len={seq_len}, d_model={d_model})")
    print(f"Output Shape         : {tuple(output.shape)}")
    print(f"Attention Map Shape  : {tuple(attn_weights.shape)} (batch, heads, seq, seq)")
    print("=" * 76)

    # Show head 1 matrix
    print_attention_table(attn_weights[0, 0], tokens, title="Head 1 Attention Weights:")

    # Show average across all heads
    avg_weights = attn_weights[0].mean(dim=0)
    print_attention_table(avg_weights, tokens, title="Averaged Attention Across All Heads:")
    print("\n")


def run_preset_questions():
    print("\n" + "=" * 76)
    print(" PRE-CONFIGURED QUESTION & BENCHMARK PROMPTS")
    print("=" * 76)
    for q in PRESET_QUESTIONS:
        mask_tag = "[Causal]" if q["causal"] else "[Bidirectional]"
        print(f"  {q['id']}. {q['title']} {mask_tag}")
        print(f"     Prompt: \"{q['sentence']}\"")
        print(f"     Focus : {q['description']}\n")

    while True:
        sel = input("Select a question number (1-5) or 'all' to run all (or 'back'): ").strip().lower()
        if sel in ("back", "exit", "quit", "q"):
            break

        if sel == "all":
            for q in PRESET_QUESTIONS:
                print("\n" + "#" * 76)
                print(f" RUNNING PRESET #{q['id']}: {q['title']}")
                print(f" Prompt: \"{q['sentence']}\"")
                print("#" * 76)
                tokens = q["sentence"].split()
                compute_and_show_attention(tokens, q["heads"], q["causal"])
            break

        try:
            choice_num = int(sel)
            selected_q = next((q for q in PRESET_QUESTIONS if q["id"] == choice_num), None)
            if not selected_q:
                print("Invalid number. Please select between 1 and 5.")
                continue

            print("\n" + "#" * 76)
            print(f" RUNNING PRESET #{selected_q['id']}: {selected_q['title']}")
            print(f" Prompt: \"{selected_q['sentence']}\"")
            print(f" Focus : {selected_q['description']}")
            print("#" * 76)

            tokens = selected_q["sentence"].split()
            compute_and_show_attention(tokens, selected_q["heads"], selected_q["causal"])

        except ValueError:
            print("Please enter a valid number (1-5), 'all', or 'back'.")


def run_text_input():
    print("\n" + "=" * 76)
    print(" 1. CUSTOM TEXT INPUT ATTENTION MODE")
    print("=" * 76)
    print("Enter a sentence to see token-by-token self-attention.\n")

    while True:
        text = input("Enter sentence (or 'back' to return to menu): ").strip()
        if text.lower() in ("back", "exit", "quit", "q"):
            break
        if not text:
            print("Please enter a non-empty string.")
            continue

        tokens = text.split()
        print(f"\nTokens detected ({len(tokens)}): {tokens}")

        mask_choice = input("Apply causal (autoregressive) mask? (y/n) [default: n]: ").strip().lower()
        use_causal = mask_choice.startswith("y")

        heads_input = input("Number of attention heads [default: 4]: ").strip()
        try:
            num_heads = int(heads_input) if heads_input else 4
        except ValueError:
            num_heads = 4

        compute_and_show_attention(tokens, num_heads, use_causal)


def run_tensor_input():
    print("\n" + "=" * 76)
    print(" 2. CUSTOM TENSOR SHAPE INPUT MODE")
    print("=" * 76)

    while True:
        try:
            batch_str = input("Enter batch size [default 1]: ").strip()
            batch_size = int(batch_str) if batch_str else 1

            seq_str = input("Enter sequence length [default 4]: ").strip()
            seq_len = int(seq_str) if seq_str else 4

            d_model_str = input("Enter d_model [default 32]: ").strip()
            d_model = int(d_model_str) if d_model_str else 32

            heads_str = input("Enter number of heads [default 4]: ").strip()
            num_heads = int(heads_str) if heads_str else 4

            if d_model % num_heads != 0:
                print(f"Error: d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
                continue

            causal_str = input("Apply causal mask? (y/n) [default n]: ").strip().lower()
            use_causal = causal_str.startswith("y")

            mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
            mha.eval()

            x = torch.randn(batch_size, seq_len, d_model)
            mask = create_causal_mask(seq_len) if use_causal else None

            with torch.no_grad():
                out, weights = mha(x, x, x, mask=mask, return_attention=True)

            print("\nExecution Results:")
            print(f"  Input  : {tuple(x.shape)}")
            print(f"  Output : {tuple(out.shape)}")
            print(f"  Weights: {tuple(weights.shape)} (batch, heads, seq, seq)")
            print("\nHead 0 Weights (Batch 0):")
            print(weights[0, 0].numpy().round(4))

        except ValueError as e:
            print(f"Invalid input: {e}")

        again = input("\nRun again? (y/n): ").strip().lower()
        if not again.startswith("y"):
            break


def main():
    while True:
        print("\n" + "=" * 76)
        print(" MODULE II - WEEK 2: ATTENTION LAB INTERACTIVE RUNNER")
        print("=" * 76)
        print("Choose an option:")
        print("  1. Enter text string / sentence (interactive tokens)")
        print("  2. Enter custom tensor dimensions (batch, seq, d_model, heads)")
        print("  3. Run Automated Tests & Verifications")
        print("  4. Choose from Pre-set Questions / Sentences")
        print("  5. Exit")

        choice = input("Select (1/2/3/4/5): ").strip()
        if choice == "1":
            run_text_input()
        elif choice == "2":
            run_tensor_input()
        elif choice == "3":
            import unittest
            from test_attention import TestAttentionMechanisms
            suite = unittest.TestLoader().loadTestsFromTestCase(TestAttentionMechanisms)
            unittest.TextTestRunner(verbosity=2).run(suite)
        elif choice == "4":
            run_preset_questions()
        elif choice in ("5", "exit", "quit", "q"):
            print("Exiting Attention Lab.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
