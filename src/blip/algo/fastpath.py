from __future__ import annotations
import re

from blip._types import Pair
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent


def _parse_sentence_ids(text: str, n: int) -> list[int]:
    """
    Parse sentence IDs from LLM-Provenance-Prompt output.
    Sentences are numbered 1..n in the prompt; convert to 0-based.
    """
    match = re.search(r"\d", text)
    if not match:
        return []
    text = text[match.start():]
    ids = re.findall(r"\d+", text)
    result = []
    seen = set()
    for id_str in ids:
        idx = int(id_str)
        # 1-indexed in prompt; filter to valid range
        if 1 <= idx <= n and idx not in seen:
            seen.add(idx)
            result.append(idx - 1)  # convert to 0-based
    return result


def elicit(pair: Pair, llm: LLMClient) -> list[int]:
    """
    Issue LLM-Provenance-Prompt and return list of 0-based sentence indices.
    Returns empty list if LLM claims no relevant sentences.
    """
    sentences = [s.text for s in sorted(pair.sentences, key=lambda s: s.idx)]
    content, _ = llm.provenance(pair.question, pair.llm_answer, sentences)
    return _parse_sentence_ids(content, len(sentences))


def verify_provenance(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
) -> tuple[bool, list]:
    """Verify that L(P, Q) ≡ A."""
    from blip.algo.prune import _text_of
    text = _text_of(sentence_idxs, pair)
    answer, usage = llm.answer(text, pair.question)
    is_equiv, judge_usages = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
    return is_equiv, [usage] + judge_usages
