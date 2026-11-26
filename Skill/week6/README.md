# Week 6: Golden Question Set with Expected Answers, Sources & Baseline Run (Release v0.1)

## Experiment Overview (CO1)
This experiment covers the foundational evaluation methodology for Generative AI applications:
1. **Curating a Golden Evaluation Dataset:** High-quality benchmark queries paired with verified canonical answers, provenance sources, and domain categories.
2. **Automated Baseline Evaluation:** Measuring lexical and semantic similarity (F1 overlap / precision / recall) and runtime latency.
3. **Release Tagging:** Freezing benchmarks at `release v0.1` to establish a performance baseline against which future prompt iterations and model upgrades are compared.

---

## Dataset Structure (`golden_dataset.json`)
Each item in the golden test suite includes:
- `id`: Unique identifier (e.g., `GQ-001`)
- `category`: Domain tag (Architecture, Reliability, Economics, etc.)
- `question`: Evaluated prompt
- `expected_answer`: Gold reference ground-truth
- `source`: Reference text / publication backing the answer

---

## How to Run

1. Open PowerShell and activate virtual environment:
```powershell
cd C:\Users\KERNEX-ACCOUTS\.gemini\antigravity\scratch\gen_ai
.\.venv\Scripts\Activate.ps1
```

2. Run the baseline evaluation against the Golden Set:
```powershell
python .\week6\baseline_evaluator.py
```

3. Run automated test suite:
```powershell
python .\week6\test_baseline.py
```

---

## Deliverables
- `golden_dataset.json`: Canonical questions, answers, and sources.
- `baseline_run_v0.1.json`: Evaluated run artifact capturing per-question F1 scores, latency, and overall pass rate for Release v0.1.

