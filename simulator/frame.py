"""
frame.py — Physical memory frames, like RAM slots.

In a real OS:
  - Virtual memory has unlimited pages
  - Physical RAM has a fixed number of frames
  - The OS maps pages → frames

In ContextOS:
  - We have unlimited document chunks (pages)
  - The LLM context window has a fixed token budget (frames)
  - We map pages → frames to fill the context window

Example:
  Frame 0 → Page "Wikipedia_Paris"
  Frame 1 → Page "Wikipedia_Eiffel_Tower"
  Frame 2 → Empty
"""
from dataclasses import dataclass, field
from typing import Optional
from .page import Page


@dataclass
class Frame:
    frame_id: int
    page: Optional[Page] = None  # None means this frame is empty

    @property
    def is_empty(self) -> bool:
        return self.page is None

    @property
    def tokens_used(self) -> int:
        return self.page.tokens if self.page else 0

    def load(self, page: Page):
        self.page = page

    def evict(self) -> Optional[Page]:
        evicted = self.page
        self.page = None
        return evicted

    def __repr__(self):
        if self.page:
            return f"Frame({self.frame_id} → {self.page.page_id})"
        return f"Frame({self.frame_id} → Empty)"
