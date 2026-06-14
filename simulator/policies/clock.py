from typing import List
from ..page import Page


class CLOCKPolicy:
    """
    Second-chance CLOCK algorithm.
    Pages with clock_bit=True get a second chance; bit is cleared.
    First page with clock_bit=False is evicted.
    """
    name = "CLOCK"

    def __init__(self):
        self._hand = 0

    def evict(self, pages: List[Page]) -> Page:
        if not pages:
            raise ValueError("No pages to evict")

        n = len(pages)
        checked = 0
        while checked < 2 * n:  # two full sweeps max
            idx = self._hand % n
            page = pages[idx]
            if not page.clock_bit:
                self._hand = (idx + 1) % n
                return page
            else:
                page.clock_bit = False
                self._hand = (idx + 1) % n
                checked += 1

        # Fallback: evict current hand position
        page = pages[self._hand % n]
        self._hand = (self._hand + 1) % n
        return page
