from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
from blip._types import Pair


@runtime_checkable
class WorkloadAdapter(Protocol):
    name: str

    def iter_pairs(self) -> Iterator[Pair]: ...

    def sentences(self, doc_id: str) -> list[str]: ...

    def tokens(self, doc_id: str) -> int: ...
