"""
asr.py — Adaptive Semantic Replacement (ASR). The novel policy in ContextOS.

Classical policies (LRU, LFU, CLOCK) are oblivious to content.
ASR scores each page on 4 dimensions before evicting:

  score = α * semantic_distance    (how irrelevant is this page to current query?)
        + β * recency_score        (how old is this page?)
        + γ * frequency_score      (how rarely was this page accessed?)
        + δ * dependency_penalty   (does anything else depend on this page?)

The page with the highest score is evicted.
A page that's semantically irrelevant, old, rarely accessed, and has no dependents
is the safest to evict — ASR finds it systematically.

This is the contribution that makes ContextOS worth posting about.
"""
import math
from typing import List, Optional
from ..page import Page


class ASRPolicy:
    name = "ASR"

    def __init__(self, alpha: float = 0.4, beta: float = 0.25,
                 gamma: float = 0.2, delta: float = 0.15):
        total = alpha + beta + gamma + delta
        assert abs(total - 1.0) < 1e-5, f"Weights must sum to 1, got {total}"
        self.alpha = alpha   # semantic distance weight
        self.beta = beta     # recency weight
        self.gamma = gamma   # frequency weight
        self.delta = delta   # dependency weight

        self.current_query_embedding: Optional[List[float]] = None
        self._all_page_ids: List[str] = []  # for dependency scoring

    def set_query(self, embedding: List[float]):
        """Call this before each reasoning step with the current query's embedding."""
        self.current_query_embedding = embedding

    def set_resident_page_ids(self, page_ids: List[str]):
        """Let ASR know all currently resident pages (for dependency scoring)."""
        self._all_page_ids = page_ids

    def evict(self, pages: List[Page]) -> Page:
        if not pages:
            raise ValueError("No pages to evict")
        if len(pages) == 1:
            return pages[0]

        semantic = self._semantic_scores(pages)
        recency  = self._recency_scores(pages)
        freq     = self._frequency_scores(pages)
        dep      = self._dependency_scores(pages)

        scores = [
            self.alpha * s + self.beta * r + self.gamma * f + self.delta * d
            for s, r, f, d in zip(semantic, recency, freq, dep)
        ]

        return pages[scores.index(max(scores))]

    # ------------------------------------------------------------------ #
    #  Scoring components                                                 #
    # ------------------------------------------------------------------ #

    def _semantic_scores(self, pages: List[Page]) -> List[float]:
        """Higher score = less semantically relevant = better eviction candidate."""
        if self.current_query_embedding is None:
            return [0.5] * len(pages)
        if any(p.embedding is None for p in pages):
            return [0.5] * len(pages)

        sims = [self._cosine(p.embedding, self.current_query_embedding) for p in pages]
        norm = self._normalize(sims)
        return [1.0 - s for s in norm]  # invert: low similarity = high eviction score

    def _recency_scores(self, pages: List[Page]) -> List[float]:
        """Higher score = accessed longer ago = better eviction candidate."""
        raw = [p.last_accessed for p in pages]
        norm = self._normalize(raw)
        return [1.0 - r for r in norm]  # invert: old = high score

    def _frequency_scores(self, pages: List[Page]) -> List[float]:
        """Higher score = accessed less often = better eviction candidate."""
        raw = [p.access_count for p in pages]
        norm = self._normalize(raw)
        return [1.0 - f for f in norm]

    def _dependency_scores(self, pages: List[Page]) -> List[float]:
        """
        Higher score = fewer things depend on this page = safer to evict.
        A page that others depend on should be harder to evict (lower score).
        """
        # Count how many resident pages list each page as a dependency
        dependent_count = {p.page_id: 0 for p in pages}
        for p in pages:
            for dep_id in p.dependencies:
                if dep_id in dependent_count:
                    # dep_id is needed by p — so dep_id has a dependent
                    # Actually we want: how many pages depend ON this page
                    pass

        # Recompute: for each page, count how many other pages list it as dependency
        is_depended_on = {p.page_id: 0 for p in pages}
        for p in pages:
            for dep_id in p.dependencies:
                if dep_id in is_depended_on:
                    is_depended_on[dep_id] += 1

        counts = [is_depended_on[p.page_id] for p in pages]
        norm = self._normalize(counts)
        return [1.0 - n for n in norm]  # fewer dependents = higher eviction score

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]
