"""Tests for Algorithm 2: Refine."""
from blip._types import Sentence, Block, Pair
from blip.algo.refine import sequential_greedy, exponential_greedy


def _make_pair(texts: list[str], answer: str = "Paris") -> Pair:
    sents = tuple(Sentence(i, t, 3) for i, t in enumerate(texts))
    return Pair(
        pair_id="t",
        doc_id="d",
        question="Q?",
        ground_truth=None,
        llm_answer=answer,
        sentences=sents,
        blocks=(),
    )


class _FakeLLM:
    def __init__(self, needed_idxs: set[int], answer: str = "Paris"):
        self._needed = needed_idxs
        self._answer = answer
        from blip.llm.usage import Usage
        self._u = Usage(100, 0, 10, "fake")

    def answer(self, text: str, question: str, model="driver"):
        # returns correct answer iff all needed sentences present
        for i in self._needed:
            if f"sent{i}" not in text:
                return "wrong", self._u
        return self._answer, self._u

    def judge(self, a: str, b: str, model=None):
        return "True" if a == b else "False", self._u


def test_seq_removes_irrelevant():
    texts = [f"sent{i}" for i in range(5)]
    pair = _make_pair(texts)
    llm = _FakeLLM(needed_idxs={0, 2})  # only sents 0 and 2 needed
    refined, usages, answer = sequential_greedy(list(range(5)), pair, llm)
    assert set(refined) == {0, 2}
    assert answer == "Paris"


def test_seq_all_needed():
    texts = [f"sent{i}" for i in range(4)]
    pair = _make_pair(texts)
    llm = _FakeLLM(needed_idxs={0, 1, 2, 3})
    refined, _, answer = sequential_greedy(list(range(4)), pair, llm)
    assert set(refined) == {0, 1, 2, 3}
    assert answer is None  # no sentence removed, no verified answer captured


def test_seq_output_sorted():
    texts = [f"sent{i}" for i in range(6)]
    pair = _make_pair(texts)
    llm = _FakeLLM(needed_idxs={1, 3, 5})
    refined, _, _ = sequential_greedy(list(range(6)), pair, llm)
    assert refined == sorted(refined)


def test_exp_removes_irrelevant():
    texts = [f"sent{i}" for i in range(5)]
    pair = _make_pair(texts)
    llm = _FakeLLM(needed_idxs={2})
    refined, _, answer = exponential_greedy(list(range(5)), pair, llm)
    assert set(refined) == {2}
    assert answer == "Paris"


def test_exp_all_needed():
    texts = [f"sent{i}" for i in range(4)]
    pair = _make_pair(texts)
    llm = _FakeLLM(needed_idxs={0, 1, 2, 3})
    refined, _, answer = exponential_greedy(list(range(4)), pair, llm)
    assert set(refined) == {0, 1, 2, 3}
    assert answer is None  # no sentence removed
