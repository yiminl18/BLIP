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
) -> tuple[list[int], list, str | None]:
    """
    Algorithm 2a: Sequential_Greedy.
    Process sentences descending for KV-cache reuse.
    Returns (refined_sentence_indices, usages, last_verified_answer).
    last_verified_answer is None if no sentence was removed (input already minimal).
    """
    p = list(sentence_idxs)
    usages = []
    last_answer: str | None = None
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
                last_answer = answer
                changed = True
    return sorted(p), usages, last_answer


def _exponential_inner(
    p: list[int],
    pair: Pair,
    llm: LLMClient,
    usages: list,
    last_answer: list,  # single-element list used as mutable ref
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
            l_exp = 0
            j -= 1
            continue
        answer, usage = llm.answer(text, pair.question)
        usages.append(usage)
        is_equiv, ju = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
        usages.extend(ju)
        if is_equiv:
            p = p_prime
            last_answer[0] = answer
            j = i - l_exp - 1 + 1  # paper line 14: j ← i − 1 − l + 1 → simplified to i - l_exp
            l_exp += 1
        else:
            l_exp = 0
            j -= 1
    return p


def exponential_greedy(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
) -> tuple[list[int], list, str | None]:
    """
    Algorithm 2b: Exponential_Greedy with outer fixed-point loop.
    Returns (refined_sentence_indices, usages, last_verified_answer).
    last_verified_answer is None if no sentence was removed.
    """
    p = list(sentence_idxs)
    usages: list = []
    last_answer: list = [None]
    p_prev = None
    while p_prev != p:
        p_prev = p[:]
        p = _exponential_inner(p, pair, llm, usages, last_answer)
    return sorted(p), usages, last_answer[0]
