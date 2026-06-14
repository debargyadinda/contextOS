from typing import List
from ..page import Page


class LFUPolicy:
    name = "LFU"

    def evict(self, pages: List[Page]) -> Page:
        """Evict the least frequently used page. Ties broken by recency."""
        return min(pages, key=lambda p: (p.access_count, p.last_accessed))
