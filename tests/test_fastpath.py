"""Tests for fast-path elicitation and sentence-ID parser."""
from blip.algo.fastpath import _parse_sentence_ids, elicit, verify_provenance
from blip._types import Sentence, Pair


def _make_pair(n: int = 10) -> Pair:
    sents = tuple(Sentence(i, f"sentence {i}", 3) for i in range(n))
    return Pair("p1", "d1", "Q?", None, "Paris", sents, ())


def test_parse_clean():
    ids = _parse_sentence_ids("3, 7, 12", 15)
    assert ids == [2, 6, 11]  # 0-based


def test_parse_bracketed():
    ids = _parse_sentence_ids("[3, 7, 12]", 15)
    assert ids == [2, 6, 11]


def test_parse_with_preamble():
    ids = _parse_sentence_ids("Sentence IDs: 3, 7, 12", 15)
    assert ids == [2, 6, 11]


def test_parse_out_of_range():
    ids = _parse_sentence_ids("0, 5, 20", 10)
    # 0 out of range (1-10), 20 out of range
    assert ids == [4]  # only 5 → 0-based 4


def test_parse_empty():
    ids = _parse_sentence_ids("None", 10)
    assert ids == []


def test_parse_newline_sep():
    ids = _parse_sentence_ids("1\n3\n5", 10)
    assert ids == [0, 2, 4]


def test_elicit_returns_ids():
    from tests.fakes.llm import FakeLLMClient
    pair = _make_pair(10)
    llm = FakeLLMClient(provenance_fn=lambda q, a, sents: "2, 5")
    ids = elicit(pair, llm)
    assert ids == [1, 4]  # 0-based


def test_verify_ok():
    from tests.fakes.llm import FakeLLMClient
    pair = _make_pair(5)
    llm = FakeLLMClient(answer_fn=lambda t, q: "Paris")
    ok, usages = verify_provenance([0, 1], pair, llm)
    assert ok is True
