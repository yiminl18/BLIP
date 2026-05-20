from __future__ import annotations
import re
import logging
from typing import Sequence

from blip._types import Block, Pair
from blip.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _parse_scores(text: str, n: int) -> list[float] | None:
    """Parse comma-separated score list. Returns None on failure."""
    # strip everything before first digit
    match = re.search(r"\d", text)
    if not match:
        return None
    text = text[match.start():]
    # extract numbers
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) < n:
        return None
    scores = [float(x) for x in nums[:n]]
    return scores


class LLMRanker:
    name = "llm"

    def __init__(self, llm: LLMClient, fallback: "EmbeddingRanker | None" = None) -> None:
        self._llm = llm
        self._fallback = fallback

    def rank(self, items: Sequence[Block], pair: Pair) -> list[int]:
        block_texts = [b.text for b in items]
        content, _ = self._llm.rank(pair.question, pair.llm_answer, block_texts)
        scores = _parse_scores(content, len(items))
        if scores is None:
            logger.warning("LLMRanker parse failure for pair %s; falling back", pair.pair_id)
            if self._fallback:
                return self._fallback.rank(items, pair)
            # uniform fallback: original order
            return list(range(len(items)))
        return sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
