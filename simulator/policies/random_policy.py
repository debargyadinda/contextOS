import random
from typing import List
from ..page import Page


class RandomPolicy:
    name = "Random"

    def evict(self, pages: List[Page]) -> Page:
        return random.choice(pages)
