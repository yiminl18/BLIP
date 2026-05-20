from __future__ import annotations
# Algorithm 3: Top-k provenance — skeleton, gated behind feature flag
# Not implemented in milestone 1.

from blip._types import Pair
from blip.llm.client import LLMClient
from blip.rank.base import Ranker


def top_k(
    pair: Pair,
    ranker: Ranker,
    llm: LLMClient,
    k: int = 3,
) -> list[list[int]]:
    raise NotImplementedError("Top-k provenance is not implemented yet (milestone 5)")
