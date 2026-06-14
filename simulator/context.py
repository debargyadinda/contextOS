"""
context.py — The core memory manager. This is the heart of ContextOS.

Simulates how an OS manages physical memory (RAM), but for LLM context.

The flow when a page is needed:
  1. Check page table → is it resident?
  2. HIT  → update access stats, return True
  3. MISS → PAGE FAULT
             → fault handler kicks in
             → if context full: choose victim → evict → update page table
             → load new page into a frame → update page table
             → return False (caller knows a fault happened)

This mirrors exactly how Linux handles a page fault.
"""
from typing import List, Optional, Dict
from .page import Page
from .frame import Frame
from .page_table import PageTable


class ContextWindow:
    def __init__(self, max_tokens: int, policy, num_frames: int = None):
        self.max_tokens = max_tokens
        self.policy = policy

        # Physical frames — like RAM slots
        # If num_frames not specified, we use token budget to determine capacity
        self._frames: List[Frame] = []
        self._next_frame_id = 0

        # Page table — tracks what's loaded where
        self.page_table = PageTable()

        # Logical clock for recency tracking
        self.clock = 0

        # Systems metrics
        self.hits = 0
        self.page_faults = 0
        self.evictions = 0
        self.eviction_log: List[str] = []
        self.access_history: List[str] = []

        # Working set window
        self.working_set_window = 5

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    @property
    def resident_pages(self) -> List[Page]:
        return [f.page for f in self._frames if not f.is_empty]

    @property
    def used_tokens(self) -> int:
        return sum(f.tokens_used for f in self._frames)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.page_faults
        return self.hits / total if total > 0 else 0.0

    @property
    def fault_rate(self) -> float:
        total = self.hits + self.page_faults
        return self.page_faults / total if total > 0 else 0.0

    @property
    def working_set(self) -> List[str]:
        return list(dict.fromkeys(self.access_history[-self.working_set_window:]))

    def request_page(self, page_id: str) -> bool:
        """
        Request access to a page.
        Returns True on hit, False on fault.
        This is the main entry point — call this whenever the LLM 'needs' a page.
        """
        self.clock += 1
        self.access_history.append(page_id)
        self.page_table.register(page_id)

        if self.page_table.is_resident(page_id):
            # HIT — update access stats
            self.hits += 1
            page = self._get_resident_page(page_id)
            if page:
                page.last_accessed = self.clock
                page.access_count += 1
                page.clock_bit = True
            return True
        else:
            # FAULT
            self.page_faults += 1
            self.page_table.mark_fault(page_id)
            return False

    def add_page(self, page: Page) -> Optional[Page]:
        """
        Load a page into context (called after a page fault to bring page in).
        Runs the fault handler if context is full.
        Returns the evicted page, or None if no eviction was needed.
        """
        self.page_table.register(page.page_id)
        evicted = None

        # Fault handler: make room if needed
        while self.used_tokens + page.tokens > self.max_tokens and self._frames:
            evicted = self._fault_handler(page)

        # Allocate a new frame and load the page
        frame = Frame(frame_id=self._next_frame_id)
        self._next_frame_id += 1
        frame.load(page)
        self._frames.append(frame)

        page.last_accessed = self.clock
        self.page_table.mark_loaded(page.page_id, frame.frame_id)

        return evicted

    def get_context_text(self) -> str:
        return "\n\n".join(p.content for p in self.resident_pages)

    def get_metrics(self) -> Dict:
        return {
            "hits": self.hits,
            "page_faults": self.page_faults,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "fault_rate": round(self.fault_rate, 4),
            "pages_in_memory": len(self.resident_pages),
            "tokens_used": self.used_tokens,
            "tokens_max": self.max_tokens,
            **self.page_table.summary(),
        }

    def reset_metrics(self):
        self.hits = 0
        self.page_faults = 0
        self.evictions = 0
        self.eviction_log = []
        self.access_history = []

    # ------------------------------------------------------------------ #
    #  Internal: fault handler                                            #
    # ------------------------------------------------------------------ #

    def _fault_handler(self, incoming_page: Page) -> Optional[Page]:
        """
        OS-style fault handler:
        1. Find evictable frames (respect dependencies of incoming page)
        2. Ask policy to choose victim
        3. Evict victim, update page table
        4. Free the frame
        """
        # Don't evict pages that the incoming page depends on
        evictable_frames = [
            f for f in self._frames
            if not f.is_empty and f.page.page_id not in incoming_page.dependencies
        ]
        if not evictable_frames:
            evictable_frames = [f for f in self._frames if not f.is_empty]

        evictable_pages = [f.page for f in evictable_frames]

        # Policy chooses the victim
        victim_page = self.policy.evict(evictable_pages)

        # Find its frame and evict
        for frame in self._frames:
            if not frame.is_empty and frame.page.page_id == victim_page.page_id:
                frame.evict()
                self._frames.remove(frame)
                self.page_table.mark_evicted(victim_page.page_id)
                self.evictions += 1
                self.eviction_log.append(victim_page.page_id)
                return victim_page

        return None

    def _get_resident_page(self, page_id: str) -> Optional[Page]:
        for f in self._frames:
            if not f.is_empty and f.page.page_id == page_id:
                return f.page
        return None

    def display_frames(self):
        """Print current frame state — useful for debugging."""
        print(f"\n{'='*40}")
        print(f"Context Window ({self.used_tokens}/{self.max_tokens} tokens)")
        print(f"{'='*40}")
        for f in self._frames:
            print(f"  {f}")
        if not self._frames:
            print("  (empty)")
        print(f"{'='*40}\n")
