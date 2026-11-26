# Week 5: Token Economics & Rate-Limit Budget for a Feature Before Building It

## Experiment Overview (CO1)
This experiment models **Token Economics** and **Rate-Limit Capacity Planning** for generative AI features prior to software development.

### Key Objectives
1. **Traffic Modeling:** Formulate mathematical models mapping Daily Active Users (DAU), queries per user, and token footprint (input context vs output generation).
2. **Cost Estimation:** Compute daily and monthly operating expenses across major frontier and lightweight models (`GPT-4o`, `GPT-4o-mini`, `Gemini 1.5 Pro/Flash`, `Claude 3.5 Sonnet`, `Local Ollama`).
3. **Rate-Limit Feasibility:** Calculate peak Requests Per Minute (RPM) and Tokens Per Minute (TPM) against API provider tier limits to prevent service degradation.
4. **Optimization Strategies:** Document architectural controls including semantic caching, prompt distillation, and model cascading.

---

## How to Run

1. Open PowerShell and activate your virtual environment:
```powershell
cd C:\Users\KERNEX-ACCOUTS\.gemini\antigravity\scratch\gen_ai
.\.venv\Scripts\Activate.ps1
```

2. Run the Token Economics model:
```powershell
python .\week5\token_economics.py
```

3. Run the automated unit tests:
```powershell
python -m unittest -v week5.test_economics
```

---

## Output Files
- `budget_report.json`: Full machine-readable audit report containing exact cost metrics and rate-limit feasibility flags.

