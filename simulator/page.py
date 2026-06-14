"""
page.py — The fundamental unit of ContextOS memory.

A Page is like a page in your OS virtual memory:
- It holds content (text chunk from a document)
- It tracks how often and how recently it was accessed
- It knows which other pages it depends on (for multi-hop reasoning)
- It has an embedding for semantic similarity (used by ASR policy)
"""
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Page:
    page_id: str
    content: str
    tokens: int = 0
    access_count: int = 0
    last_accessed: int = 0       # logical clock tick — higher = more recent
    clock_bit: bool = False      # used by CLOCK algorithm (second-chance bit)
    embedding: Any = None        # vector embedding for ASR semantic scoring
    dependencies: List[str] = field(default_factory=list)  # page_ids this page needs to reason

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = max(1, len(self.content) // 4)  # ~4 chars per token

    def __repr__(self):
        return f"Page(id={self.page_id}, tokens={self.tokens}, accesses={self.access_count})"
