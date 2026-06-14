"""
page_table.py — The mapping between pages and physical frames.

In a real OS, the page table answers:
  "Is page X in RAM? If yes, which frame?"

In ContextOS, it answers:
  "Is chunk X currently in the LLM context? If yes, which frame?"

It also tracks per-page fault history so we can analyze
which pages cause the most faults across a benchmark run.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PageTableEntry:
    page_id: str
    resident: bool = False      # True = currently loaded in a frame
    frame_id: Optional[int] = None
    fault_count: int = 0        # how many times this page caused a fault
    load_count: int = 0         # how many times this page was loaded


class PageTable:
    def __init__(self):
        self._table: Dict[str, PageTableEntry] = {}

    def register(self, page_id: str):
        """Register a page (called when we first see it)."""
        if page_id not in self._table:
            self._table[page_id] = PageTableEntry(page_id=page_id)

    def mark_loaded(self, page_id: str, frame_id: int):
        self.register(page_id)
        entry = self._table[page_id]
        entry.resident = True
        entry.frame_id = frame_id
        entry.load_count += 1

    def mark_evicted(self, page_id: str):
        if page_id in self._table:
            self._table[page_id].resident = False
            self._table[page_id].frame_id = None

    def mark_fault(self, page_id: str):
        self.register(page_id)
        self._table[page_id].fault_count += 1

    def is_resident(self, page_id: str) -> bool:
        return self._table.get(page_id, PageTableEntry(page_id)).resident

    def get_frame(self, page_id: str) -> Optional[int]:
        entry = self._table.get(page_id)
        return entry.frame_id if entry else None

    def get_entry(self, page_id: str) -> Optional[PageTableEntry]:
        return self._table.get(page_id)

    def summary(self) -> Dict:
        total = len(self._table)
        resident = sum(1 for e in self._table.values() if e.resident)
        total_faults = sum(e.fault_count for e in self._table.values())
        return {
            "total_pages_seen": total,
            "currently_resident": resident,
            "total_faults_logged": total_faults,
        }
