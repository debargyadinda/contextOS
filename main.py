"""
main.py — Entry point for ContextOS benchmark.

Usage:
  # Dry run (no API key, systems metrics only):
  python -m contextos.main

  # Full run with LLM scoring:
  set GEMINI_API_KEY=your-key   (Windows)
  export GEMINI_API_KEY=your-key (Linux/Mac)
  python -m contextos.main --samples 20

  # Memory pressure experiment (the main result):
  python -m contextos.main --mode pressure --samples 30 --save
"""
import os
import json
import argparse
from pathlib import Path

from contextos.simulator.policies import ALL_POLICIES
from contextos.benchmark.runner import run_full_benchmark, run_memory_pressure, DEFAULT_BUDGETS
from contextos.results.plots import print_table, plot_policy_comparison, plot_memory_pressure


def main():
    parser = argparse.ArgumentParser(description="ContextOS Benchmark")
    parser.add_argument("--mode", choices=["standard", "pressure"], default="standard",
                        help="standard = one token budget | pressure = multiple budgets")
    parser.add_argument("--samples",    type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Token budget for standard mode (default: 150)")
    parser.add_argument("--output",     type=str, default="results")
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found — running in dry mode (systems metrics only).")
        print("Set GEMINI_API_KEY to enable LLM reasoning accuracy scoring.\n")

    Path(args.output).mkdir(exist_ok=True)

    if args.mode == "standard":
        results = run_full_benchmark(
            ALL_POLICIES, n_samples=args.samples,
            max_tokens=args.max_tokens, api_key=api_key
        )
        print_table(results)
        plot_policy_comparison(results, args.output)

    elif args.mode == "pressure":
        results = run_memory_pressure(
            ALL_POLICIES, n_samples=args.samples,
            token_budgets=DEFAULT_BUDGETS,
            api_key=api_key
        )
        plot_memory_pressure(results, args.output)

    if args.save:
        out = f"{args.output}/results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved raw results: {out}")


if __name__ == "__main__":
    main()