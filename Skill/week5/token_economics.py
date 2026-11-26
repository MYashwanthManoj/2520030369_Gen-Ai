import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# Pricing per 1 Million tokens (Standard industry benchmark)
# Format: model_name -> (input_cost_per_1M, output_cost_per_1M, tpm_limit, rpm_limit)
PRICING_CATALOG = {
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "rpm_limit": 500,
        "tpm_limit": 30000,
        "description": "OpenAI flagship multi-modal model"
    },
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "rpm_limit": 5000,
        "tpm_limit": 200000,
        "description": "Fast and cheap lightweight reasoning model"
    },
    "claude-3-5-sonnet": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "rpm_limit": 1000,
        "tpm_limit": 40000,
        "description": "Anthropic high-capability coding & reasoning model"
    },
    "gemini-1.5-flash": {
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "rpm_limit": 2000,
        "tpm_limit": 4000000,
        "description": "Google high-speed cost-efficient long-context model"
    },
    "gemini-1.5-pro": {
        "input_per_1m": 1.25,
        "output_per_1m": 5.00,
        "rpm_limit": 360,
        "tpm_limit": 2000000,
        "description": "Google flagship multimodal reasoning model"
    },
    "ollama-local (llama3/mistral)": {
        "input_per_1m": 0.00,
        "output_per_1m": 0.00,
        "rpm_limit": 60,  # hardware bounded
        "tpm_limit": 5000, # local GPU/CPU compute bound
        "description": "Self-hosted local open-weights model"
    }
}

@dataclass
class FeatureTrafficSpec:
    feature_name: str
    expected_daily_active_users: int
    queries_per_user_per_day: int
    avg_input_tokens_per_call: int
    avg_output_tokens_per_call: int
    peak_concurrency_multiplier: float = 2.5 # Peak traffic vs avg traffic

class TokenEconomicsCalculator:
    def __init__(self, spec: FeatureTrafficSpec):
        self.spec = spec
        self.daily_calls = spec.expected_daily_active_users * spec.queries_per_user_per_day
        self.monthly_calls = self.daily_calls * 30
        self.daily_input_tokens = self.daily_calls * spec.avg_input_tokens_per_call
        self.daily_output_tokens = self.daily_calls * spec.avg_output_tokens_per_call
        self.total_daily_tokens = self.daily_input_tokens + self.daily_output_tokens
        
        # Rate limit estimation:
        # Assuming traffic happens across an 8-hour active window per day:
        active_seconds = 8 * 3600
        avg_calls_per_sec = self.daily_calls / active_seconds
        peak_calls_per_min = avg_calls_per_sec * spec.peak_concurrency_multiplier * 60
        peak_tokens_per_min = peak_calls_per_min * (spec.avg_input_tokens_per_call + spec.avg_output_tokens_per_call)
        
        self.peak_rpm_needed = round(peak_calls_per_min, 1)
        self.peak_tpm_needed = round(peak_tokens_per_min, 1)

    def calculate_cost_for_model(self, model_key: str) -> Dict[str, Any]:
        info = PRICING_CATALOG.get(model_key)
        if not info:
            raise ValueError(f"Unknown model: {model_key}")
        
        input_cost_daily = (self.daily_input_tokens / 1_000_000) * info["input_per_1m"]
        output_cost_daily = (self.daily_output_tokens / 1_000_000) * info["output_per_1m"]
        daily_total = input_cost_daily + output_cost_daily
        monthly_total = daily_total * 30

        # Rate limit viability check
        rpm_exceeded = self.peak_rpm_needed > info["rpm_limit"]
        tpm_exceeded = self.peak_tpm_needed > info["tpm_limit"]
        rate_limit_status = "VIABLE"
        if rpm_exceeded or tpm_exceeded:
            rate_limit_status = "TIER_UPGRADE_OR_RATE_LIMITER_REQUIRED"

        return {
            "model": model_key,
            "daily_input_cost_usd": round(input_cost_daily, 4),
            "daily_output_cost_usd": round(output_cost_daily, 4),
            "daily_total_cost_usd": round(daily_total, 4),
            "monthly_total_cost_usd": round(monthly_total, 2),
            "model_rpm_limit": info["rpm_limit"],
            "model_tpm_limit": info["tpm_limit"],
            "peak_rpm_needed": self.peak_rpm_needed,
            "peak_tpm_needed": self.peak_tpm_needed,
            "rate_limit_status": rate_limit_status,
            "description": info["description"]
        }

    def generate_full_comparison_report(self) -> Dict[str, Any]:
        results = []
        for model in PRICING_CATALOG.keys():
            results.append(self.calculate_cost_for_model(model))
        return {
            "feature_spec": asdict(self.spec),
            "daily_volume": {
                "daily_calls": self.daily_calls,
                "monthly_calls": self.monthly_calls,
                "daily_input_tokens": self.daily_input_tokens,
                "daily_output_tokens": self.daily_output_tokens,
                "total_daily_tokens": self.total_daily_tokens
            },
            "rate_limit_requirements": {
                "peak_rpm_needed": self.peak_rpm_needed,
                "peak_tpm_needed": self.peak_tpm_needed
            },
            "model_cost_evaluations": results
        }


def print_formatted_report(report: Dict[str, Any]):
    spec = report["feature_spec"]
    vol = report["daily_volume"]
    req = report["rate_limit_requirements"]

    print("=" * 80)
    print(f" EXPERIMENT 5: TOKEN ECONOMICS & RATE-LIMIT BUDGET ANALYSIS")
    print(f" Feature Name: '{spec['feature_name']}'")
    print("=" * 80)
    print(f" [Traffic Assumptions]")
    print(f"  * Expected DAU (Daily Active Users): {spec['expected_daily_active_users']:,}")
    print(f"  * Queries per User/Day:              {spec['queries_per_user_per_day']}")
    print(f"  * Avg Input Tokens/Call:             {spec['avg_input_tokens_per_call']:,} tokens (Context / Prompt)")
    print(f"  * Avg Output Tokens/Call:            {spec['avg_output_tokens_per_call']:,} tokens (LLM Completion)")
    print(f"  * Total Daily LLM Calls:             {vol['daily_calls']:,} calls/day ({vol['monthly_calls']:,}/month)")
    print(f"  * Total Daily Token Consumption:     {vol['total_daily_tokens']:,} tokens/day")
    print("-" * 80)
    print(f" [Rate-Limit Demand (Peak 8-hour window)]")
    print(f"  * Peak RPM Demand:                   {req['peak_rpm_needed']} Requests/min")
    print(f"  * Peak TPM Demand:                   {req['peak_tpm_needed']:,} Tokens/min")
    print("=" * 80)
    print(f"{'Model':<28} | {'Daily ($)':<10} | {'Monthly ($)':<12} | {'Rate Status':<15}")
    print("-" * 80)

    for item in report["model_cost_evaluations"]:
        print(f"{item['model']:<28} | ${item['daily_total_cost_usd']:<9.2f} | ${item['monthly_total_cost_usd']:<11.2f} | {item['rate_limit_status']}")

    print("=" * 80)
    print(" Architectural Recommendations for Cost & Rate Control:")
    print(" 1. Semantic Caching (Redis/In-memory): Reduces input tokens by ~25-40% on recurring prompts.")
    print(" 2. Prompt Optimization: Trim unneeded system prompt context to reduce input token overhead.")
    print(" 3. Tiered Model Fallback: Route simple classification queries to gpt-4o-mini / gemini-flash,")
    print("    and complex multi-step reasoning to gpt-4o / gemini-pro.")
    print(" 4. Client-side Rate Limiting: Implement Token Bucket / Leaky Bucket algorithms before hitting LLM API.")
    print("=" * 80)


def main():
    # Example Feature: AI Customer Support & Document Assistant
    feature = FeatureTrafficSpec(
        feature_name="Smart Document Search & QnA Chatbot",
        expected_daily_active_users=1500,
        queries_per_user_per_day=4,
        avg_input_tokens_per_call=1200,   # Prompt with retrieved RAG context
        avg_output_tokens_per_call=350,   # Detailed answer
        peak_concurrency_multiplier=2.5
    )

    calculator = TokenEconomicsCalculator(feature)
    report = calculator.generate_full_comparison_report()
    
    # Save the output to JSON for documentation / grading audit
    output_path = os.path.join(os.path.dirname(__file__), "budget_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print_formatted_report(report)
    print(f"[SUCCESS] Detailed budget analysis report exported to: budget_report.json\n")


if __name__ == "__main__":
    main()

