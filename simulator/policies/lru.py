from typing import List
from ..page import Page


class LRUPolicy:
    name = "LRU"

    def evict(self, pages: List[Page]) -> Page:
        """Evict the least recently used page."""
        return min(pages, key=lambda p: p.last_accessed)
