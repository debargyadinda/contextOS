"""
plots.py — Generates the benchmark charts for your LinkedIn post.

Three charts:
1. Policy comparison bar chart (accuracy + fault rate at one token budget)
2. Memory pressure curves (accuracy vs token budget, one line per policy)
3. Accuracy vs fault rate scatter (the money shot — lower-right = best)
"""
from typing import List, Dict
from pathlib import Path


def _check_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError:
        print("Run: pip install matplotlib numpy")
        return None, None


def print_table(results: List[Dict]):
    """Print a clean comparison table to the terminal."""
    # Filter to single-budget results
    single = [r for r in results if "accuracy_em" in r]
    if not single:
        return

    print("\n" + "=" * 70)
    print(f"{'Policy':<15} {'EM %':>7} {'F1 %':>7} {'Hit %':>9} {'Fault %':>9} {'Evictions':>10}")
    print("-" * 70)
    for r in sorted(single, key=lambda x: x["accuracy_em"], reverse=True):
        print(f"{r['policy']:<15} "
              f"{r['accuracy_em']*100:>6.1f}% "
              f"{r['accuracy_f1']*100:>6.1f}% "
              f"{r['avg_hit_rate']*100:>8.1f}% "
              f"{r['avg_fault_rate']*100:>8.1f}% "
              f"{r['total_evictions']:>10}")
    print("=" * 70)


def plot_policy_comparison(results: List[Dict], output_dir: str = "results"):
    plt, np = _check_matplotlib()
    if plt is None:
        return

    Path(output_dir).mkdir(exist_ok=True)
    policies = [r["policy"] for r in results]
    em      = [r["accuracy_em"] * 100 for r in results]
    fault   = [r["avg_fault_rate"] * 100 for r in results]
    hit     = [r["avg_hit_rate"] * 100 for r in results]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    x = np.arange(len(policies))
    w = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ContextOS — Page Replacement Policy Benchmark (HotpotQA)",
                 fontsize=13, fontweight="bold")

    # Chart 1: Accuracy
    bars = axes[0].bar(x, em, color=colors[:len(policies)], width=w*2, alpha=0.85)
    axes[0].set_title("Reasoning Accuracy (Exact Match %)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(policies, rotation=20)
    axes[0].set_ylabel("%"); axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, em):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=9)

    # Chart 2: Hit vs Fault
    axes[1].bar(x, hit,   color=colors[:len(policies)], width=w*2, alpha=0.85, label="Hit Rate")
    axes[1].bar(x, fault, color=colors[:len(policies)], width=w*2, alpha=0.35,
                bottom=hit, label="Fault Rate")
    axes[1].set_title("Hit Rate vs Fault Rate")
    axes[1].set_xticks(x); axes[1].set_xticklabels(policies, rotation=20)
    axes[1].set_ylabel("%"); axes[1].legend()

    # Chart 3: Scatter — lower-right is best
    for i, (p, e, f) in enumerate(zip(policies, em, fault)):
        axes[2].scatter(f, e, s=140, color=colors[i], zorder=5, label=p)
        axes[2].annotate(p, (f, e), textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes[2].set_xlabel("Fault Rate (%) ↓ lower is better")
    axes[2].set_ylabel("Accuracy (%) ↑ higher is better")
    axes[2].set_title("Accuracy vs Fault Rate\n(lower-right corner = best)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = f"{output_dir}/policy_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_memory_pressure(results: List[Dict], output_dir: str = "results"):
    """Plot accuracy and fault rate curves across token budgets."""
    plt, np = _check_matplotlib()
    if plt is None:
        return

    Path(output_dir).mkdir(exist_ok=True)

    # Group by policy
    from collections import defaultdict
    by_policy = defaultdict(list)
    for r in sorted(results, key=lambda x: x["max_tokens"]):
        by_policy[r["policy"]].append(r)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    policy_names = list(by_policy.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ContextOS — Memory Pressure Experiment (HotpotQA)",
                 fontsize=13, fontweight="bold")

    for i, pname in enumerate(policy_names):
        data = by_policy[pname]
        budgets = [d["max_tokens"] for d in data]
        acc     = [d["accuracy_em"] * 100 for d in data]
        fault   = [d["avg_fault_rate"] * 100 for d in data]
        c = colors[i % len(colors)]

        axes[0].plot(budgets, acc,   marker="o", color=c, label=pname, linewidth=2)
        axes[1].plot(budgets, fault, marker="o", color=c, label=pname, linewidth=2)

    axes[0].set_title("Accuracy vs Context Budget")
    axes[0].set_xlabel("Token Budget"); axes[0].set_ylabel("Exact Match %")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Fault Rate vs Context Budget")
    axes[1].set_xlabel("Token Budget"); axes[1].set_ylabel("Fault Rate %")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{output_dir}/memory_pressure.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
