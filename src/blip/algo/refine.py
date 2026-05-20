from __future__ import annotations
from blip._types import Pair
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent


def _text_of(sentence_idxs: list[int], pair: Pair) -> str:
    sent_map = {s.idx: s.text for s in pair.sentences}
    return " ".join(sent_map[i] for i in sorted(sentence_idxs))


def sequential_greedy(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
) -> tuple[list[int], list]:
    """
    Algorithm 2a: Sequential_Greedy.
    Process sentences descending for KV-cache reuse.
    Returns (refined_sentence_indices, usages).
    """
    p = list(sentence_idxs)
    usages = []
    changed = True
    while changed:
        changed = False
        for s in sorted(p, reverse=True):  # descending index → max KV reuse
            p_prime = [x for x in p if x != s]
            if not p_prime:
                continue
            text = _text_of(p_prime, pair)
            answer, usage = llm.answer(text, pair.question)
            usages.append(usage)
            is_equiv, ju = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
            usages.extend(ju)
            if is_equiv:
                p = p_prime
                changed = True
    return sorted(p), usages


def _exponential_inner(
    p: list[int],
    pair: Pair,
    llm: LLMClient,
    usages: list,
) -> list[int]:
    """One pass of exponential_greedy_inner."""
    j = len(p) - 1
    l_exp = 0
    while j >= 0:
        i = max(0, j - (2 ** l_exp) + 1)
        chunk = set(p[i:j + 1])
        p_prime = [x for x in p if x not in chunk]
        text = _text_of(p_prime, pair) if p_prime else ""
        if not p_prime:
            # can't remove all sentences
            l_exp = 0
            j -= 1
            continue
        answer, usage = llm.answer(text, pair.question)
        usages.append(usage)
        is_equiv, ju = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
        usages.extend(ju)
        if is_equiv:
            p = p_prime
            j = i - l_exp - 1 + 1  # paper line 14: j ← i − 1 − l + 1 → simplified to i - l_exp
            l_exp += 1
        else:
            # reset; j stays same, retry with window=1 next iter
            l_exp = 0
            j -= 1
    return p


def exponential_greedy(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
) -> tuple[list[int], list]:
    """
    Algorithm 2b: Exponential_Greedy with outer fixed-point loop.
    Returns (refined_sentence_indices, usages).
    """
    p = list(sentence_idxs)
    usages: list = []
    p_prev = None
    while p_prev != p:
        p_prev = p[:]
        p = _exponential_inner(p, pair, llm, usages)
    return sorted(p), usages
