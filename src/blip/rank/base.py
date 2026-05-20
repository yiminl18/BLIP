from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
from blip._types import Block, Pair


@runtime_checkable
class Ranker(Protocol):
    name: str

    def rank(self, items: Sequence[Block], pair: Pair) -> list[int]:
        """Return block indices in descending relevance order."""
        ...
