from typing import List, Set
from ..page import Page


class WorkingSetPolicy:
    """
    Working Set replacement policy.
    Evicts pages NOT in the current working set (pages accessed in last N steps).
    Falls back to LRU if all pages are in the working set.
    """
    name = "WorkingSet"

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.access_history: List[str] = []

    def record_access(self, page_id: str):
        self.access_history.append(page_id)

    @property
    def working_set(self) -> Set[str]:
        return set(self.access_history[-self.window_size:])

    def evict(self, pages: List[Page]) -> Page:
        ws = self.working_set
        outside_ws = [p for p in pages if p.page_id not in ws]

        if outside_ws:
            # Evict LRU among pages outside working set
            return min(outside_ws, key=lambda p: p.last_accessed)
        else:
            # All pages in working set — fall back to LRU
            return min(pages, key=lambda p: p.last_accessed)
