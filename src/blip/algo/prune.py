from __future__ import annotations
import math
from typing import Literal

from blip._types import Pair, Sentence
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent
from blip.rank.base import Ranker


def _text_of(sentence_idxs: list[int], pair: Pair) -> str:
    sent_map = {s.idx: s.text for s in pair.sentences}
    return " ".join(sent_map[i] for i in sorted(sentence_idxs))


def _block_idxs_to_sentence_idxs(block_indices: list[int], pair: Pair) -> list[int]:
    """Expand block indices to sorted sentence indices."""
    result = []
    for bi in block_indices:
        result.extend(pair.blocks[bi].sentence_idxs)
    return sorted(set(result))


def prune_bottom_up(
    pair: Pair,
    ranker: Ranker,
    llm: LLMClient,
) -> tuple[list[int], list]:
    """
    Algorithm 1 bottom-up scan.
    Returns (sentence_indices, usages).
    """
    order = ranker.rank(pair.blocks, pair)  # block indices, most-relevant first
    accumulated: list[int] = []  # block indices
    usages = []

    for block_idx in order:
        accumulated.append(block_idx)
        sent_idxs = _block_idxs_to_sentence_idxs(accumulated, pair)
        text = _text_of(sent_idxs, pair)
        answer, usage = llm.answer(text, pair.question)
        usages.append(usage)
        is_equiv, judge_usages = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
        usages.extend(judge_usages)
        if is_equiv:
            break

    return _block_idxs_to_sentence_idxs(accumulated, pair), usages


def prune_top_down(
    pair: Pair,
    ranker: Ranker,
    llm: LLMClient,
) -> tuple[list[int], list]:
    """
    Algorithm 1 top-down binary search.
    Returns (sentence_indices, usages).
    """
    order = ranker.rank(pair.blocks, pair)
    m = len(pair.blocks)
    usages = []

    # sanity: full T must verify
    full_sent_idxs = _block_idxs_to_sentence_idxs(list(range(m)), pair)
    full_text = _text_of(full_sent_idxs, pair)
    answer_full, usage_full = llm.answer(full_text, pair.question)
    usages.append(usage_full)
    is_equiv_full, ju = equivalent(answer_full, pair.llm_answer, llm_client=llm, pair=pair)
    usages.extend(ju)
    assert is_equiv_full, f"Full text does not verify for pair {pair.pair_id}"

    l, r = 1, m
    best: list[int] = full_sent_idxs

    while l <= r:
        mid = math.ceil((l + r) / 2)
        candidate_blocks = order[:mid]
        candidate_sent = _block_idxs_to_sentence_idxs(candidate_blocks, pair)
        text = _text_of(candidate_sent, pair)
        answer, usage = llm.answer(text, pair.question)
        usages.append(usage)
        is_equiv, ju = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
        usages.extend(ju)
        if is_equiv:
            best = candidate_sent
            r = mid - 1
        else:
            l = mid + 1

    return best, usages


def prune(
    pair: Pair,
    ranker: Ranker,
    llm: LLMClient,
    scan: Literal["bottom_up", "top_down"] = "bottom_up",
) -> tuple[list[int], list]:
    if scan == "bottom_up":
        return prune_bottom_up(pair, ranker, llm)
    elif scan == "top_down":
        return prune_top_down(pair, ranker, llm)
    else:
        raise ValueError(f"Unknown scan: {scan}")
