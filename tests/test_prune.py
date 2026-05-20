"""Tests for Algorithm 1: Prune."""
import pytest
from blip._types import Sentence, Block, Pair
from blip.algo.prune import prune_bottom_up, prune_top_down
from blip.rank.embedding import EmbeddingRanker
from tests.fakes.llm import FakeLLMClient


def _make_pair(n_sents: int = 20) -> Pair:
    sents = tuple(Sentence(i, f"sentence {i}", 3) for i in range(n_sents))
    # build 4 blocks of 5 sentences each
    m = 4
    size = n_sents // m
    blocks = []
    for b in range(m):
        idxs = tuple(range(b * size, (b + 1) * size))
        text = " ".join(f"sentence {i}" for i in idxs)
        blocks.append(Block(idx=b, sentence_idxs=idxs, text=text, token_count=size * 3))
    return Pair(
        pair_id="test-001",
        doc_id="doc-001",
        question="What is X?",
        ground_truth=None,
        llm_answer="Paris",
        sentences=sents,
        blocks=tuple(blocks),
    )


class FixedRanker:
    name = "fixed"
    def __init__(self, order: list[int]) -> None:
        self._order = order
    def rank(self, items, pair) -> list[int]:
        return self._order[:len(items)]


def test_bottom_up_stops_early():
    """LLM returns correct answer on first block → only 1 answer call."""
    pair = _make_pair(20)
    # answer always matches
    llm = FakeLLMClient(answer_fn=lambda text, q: "Paris")
    ranker = FixedRanker([0, 1, 2, 3])
    idxs, usages = prune_bottom_up(pair, ranker, llm)
    # should stop after block 0 (5 sentences)
    assert len(idxs) == 5
    # 1 answer call + 1 judge call
    assert len(llm.answer_calls) == 1


def test_bottom_up_all_blocks():
    """LLM never matches until all blocks are added."""
    pair = _make_pair(20)
    call_count = [0]

    def answer_fn(text, q):
        call_count[0] += 1
        if "sentence 19" in text:
            return "Paris"
        return "wrong"

    # judge returns True only when answer matches target
    def judge_fn(a, b):
        return "True" if a == b else "False"

    llm = FakeLLMClient(answer_fn=answer_fn, judge_fn=judge_fn)
    ranker = FixedRanker([0, 1, 2, 3])
    idxs, usages = prune_bottom_up(pair, ranker, llm)
    assert sorted(idxs) == list(range(20))
    assert call_count[0] == 4  # one per block


def test_top_down_binary_search():
    """Top-down with answer matching on first 2 blocks."""
    pair = _make_pair(20)

    def answer_fn(text, q):
        # match if sentences 0-9 present (first 2 blocks)
        if "sentence 9" in text and "sentence 0" in text:
            return "Paris"
        if "sentence 0" in text and "sentence 4" in text and "sentence 9" not in text:
            return "wrong"
        return "Paris"  # full text

    llm = FakeLLMClient(answer_fn=answer_fn)
    ranker = FixedRanker([0, 1, 2, 3])
    idxs, usages = prune_top_down(pair, ranker, llm)
    # should find a minimal passing set
    assert len(idxs) > 0
    assert all(isinstance(i, int) for i in idxs)


def test_bottom_up_reorders_to_t_order():
    """Provenance indices must always be in ascending T order."""
    pair = _make_pair(20)
    llm = FakeLLMClient(answer_fn=lambda text, q: "Paris")
    ranker = FixedRanker([3, 2, 1, 0])  # reverse order
    idxs, _ = prune_bottom_up(pair, ranker, llm)
    assert idxs == sorted(idxs)
