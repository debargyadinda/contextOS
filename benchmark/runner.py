"""
runner.py — Runs the benchmark across all policies and token budgets.

Two modes:
1. Dry run (no API key): measures only systems metrics (hit rate, fault rate, evictions)
2. Full run (with API key): also measures reasoning accuracy (exact match, F1)

Memory pressure experiments:
  Run each policy at multiple token budgets.
  Shows how hit rate and fault rate change as context gets tighter.
"""
import json
import time
import random
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from contextos.simulator.context import ContextWindow
from contextos.simulator.page import Page
from contextos.benchmark.dataset import download_hotpotqa, sample_to_pages
from contextos.benchmark.scorer import exact_match, token_f1
from contextos.simulator.policies.asr import ASRPolicy
from contextos.simulator.policies.working_set import WorkingSetPolicy

_GEMINI_SLEEP = 32
_MAX_RETRIES  = 5

# Token budgets calibrated to HotpotQA page sizes (~30-50 tokens/page, 10 pages/sample)
# This ensures evictions actually happen across all budget levels
DEFAULT_BUDGETS = [60, 100, 150, 200, 300]


def query_llm(context_text: str, question: str, api_key: str) -> str:
    prompt = (
        f"Answer using only the context below. Be as brief as possible.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 64, "temperature": 0.0}
    }).encode()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = _GEMINI_SLEEP * (2 ** attempt)
                print(f"      429 rate-limited — waiting {wait}s (attempt {attempt+1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES} retries")


def _notify_policy(policy, page_id: str, query_embedding, resident_pages=None):
    """Feed policy-specific state after each access."""
    if isinstance(policy, WorkingSetPolicy):
        policy.record_access(page_id)
    if isinstance(policy, ASRPolicy):
        if query_embedding is not None:
            policy.set_query(query_embedding)
        if resident_pages is not None:
            policy.set_resident_page_ids([p.page_id for p in resident_pages])


def _simulate_access(ctx: ContextWindow, pages: List[Page],
                     policy, query_embedding, n_rounds: int = 3):
    """
    Simulate multi-round reasoning: the agent makes N passes over pages,
    accessing them in random order each round. This ensures the context
    fills up and evictions happen even at higher budgets.
    """
    for _ in range(n_rounds):
        order = pages[:]
        random.shuffle(order)
        for page in order:
            hit = ctx.request_page(page.page_id)
            _notify_policy(policy, page.page_id, query_embedding,
                           resident_pages=ctx.resident_pages)
            if not hit:
                ctx.add_page(page)


def run_one_policy(policy, samples: List[Dict], max_tokens: int,
                   api_key: Optional[str] = None) -> Dict:
    results = []

    for i, sample in enumerate(samples):
        pages, question, answer, query_embedding = sample_to_pages(sample)

        ctx = ContextWindow(max_tokens=max_tokens, policy=policy)

        # Prime ASR with query embedding
        if isinstance(policy, ASRPolicy) and query_embedding is not None:
            policy.set_query(query_embedding)

        # Load first 2 pages to start
        for page in pages[:2]:
            ctx.add_page(page)
            _notify_policy(policy, page.page_id, query_embedding,
                           resident_pages=ctx.resident_pages)

        # Multi-round access simulation — this is what forces evictions
        _simulate_access(ctx, pages, policy, query_embedding, n_rounds=3)

        context_text = ctx.get_context_text()

        if api_key:
            try:
                time.sleep(_GEMINI_SLEEP)
                prediction = query_llm(context_text, question, api_key)
            except Exception as e:
                print(f"    API error on sample {i}: {e}")
                prediction = "unknown"
        else:
            prediction = "unknown"

        em = exact_match(prediction, answer)
        f1 = token_f1(prediction, answer)

        results.append({
            "exact_match": em,
            "f1": f1,
            **ctx.get_metrics(),
        })

        if (i + 1) % 5 == 0:
            print(f"    [{policy.name}] {i+1}/{len(samples)}")

    n = len(results)
    return {
        "policy":          policy.name,
        "max_tokens":      max_tokens,
        "samples":         n,
        "accuracy_em":     round(sum(r["exact_match"] for r in results) / n, 4),
        "accuracy_f1":     round(sum(r["f1"] for r in results) / n, 4),
        "avg_hit_rate":    round(sum(r["hit_rate"] for r in results) / n, 4),
        "avg_fault_rate":  round(sum(r["fault_rate"] for r in results) / n, 4),
        "total_evictions": sum(r["evictions"] for r in results),
    }


def run_full_benchmark(policies, n_samples=50, max_tokens=150,
                       api_key=None) -> List[Dict]:
    print(f"Loading {n_samples} HotpotQA samples...")
    samples = download_hotpotqa(max_samples=n_samples)

    results = []
    for policy in policies:
        print(f"\nRunning: {policy.name}")
        r = run_one_policy(policy, samples, max_tokens, api_key)
        results.append(r)
        print(f"  EM={r['accuracy_em']} F1={r['accuracy_f1']} "
              f"Hit={r['avg_hit_rate']} Fault={r['avg_fault_rate']}")
    return results


def run_memory_pressure(policies, n_samples=30,
                        token_budgets=None, api_key=None) -> List[Dict]:
    """Memory pressure: each policy at multiple token budgets."""
    if token_budgets is None:
        token_budgets = DEFAULT_BUDGETS

    print(f"Loading {n_samples} HotpotQA samples...")
    samples = download_hotpotqa(max_samples=n_samples)

    results = []
    for budget in token_budgets:
        print(f"\n--- Token budget: {budget} ---")
        for policy in policies:
            print(f"  {policy.name}...")
            r = run_one_policy(policy, samples, budget, api_key)
            results.append(r)
            print(f"    Fault={r['avg_fault_rate']:.4f} Hit={r['avg_hit_rate']:.4f} Evictions={r['total_evictions']}")

    return results