from __future__ import annotations
import math

from blip._types import Pair
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent
from blip.rank.base import Ranker
from blip.algo.prune import prune_top_down, _block_idxs_to_sentence_idxs, _text_of

# Crossover point for m=20 (from paper §3.3.2 Theorem 4)
_CP_DEFAULT = 8


def _compute_cp(m: int) -> int:
    L_m = (math.sqrt(8 * m - 7) - 1) / 2
    U_m = (-1 + math.sqrt(1 + 8 * (m * math.log2(m) - m + 1))) / 2
    return round((L_m + U_m) / 2)


def adaptive_prune(
    pair: Pair,
    ranker: Ranker,
    llm: LLMClient,
    cp: int = _CP_DEFAULT,
) -> tuple[list[int], list]:
    """
    Adaptive prune: bottom-up over top-CP blocks, then top-down if no hit.
    Returns (sentence_indices, usages).
    """
    order = ranker.rank(pair.blocks, pair)
    usages = []

    # Phase: bottom-up over top-cp blocks
    accumulated: list[int] = []
    for block_idx in order[:cp]:
        accumulated.append(block_idx)
        sent_idxs = _block_idxs_to_sentence_idxs(accumulated, pair)
        text = _text_of(sent_idxs, pair)
        answer, usage = llm.answer(text, pair.question)
        usages.append(usage)
        is_equiv, ju = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
        usages.extend(ju)
        if is_equiv:
            return sorted(sent_idxs), usages

    # Bottom-up exhausted CP blocks without success → top-down over all m
    td_idxs, td_usages = prune_top_down(pair, ranker, llm)
    usages.extend(td_usages)
    return td_idxs, usages
