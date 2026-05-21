"""Tests for text processing: segmenter, blocks, tokens."""
from blip.text.segmenter import segment
from blip.text.blocks import build_blocks
from blip.text.tokens import count_tokens
from blip._types import Sentence


def test_segment_basic():
    text = "This is sentence one. This is sentence two. And three."
    sents = segment(text, min_tokens=1)  # no merging
    assert len(sents) >= 2
    assert all(isinstance(s, str) for s in sents)


def test_segment_nonempty():
    sents = segment("Hello world.", min_tokens=1)
    assert len(sents) == 1
    assert sents[0] == "Hello world."


def test_blocks_equal_split():
    sents = [Sentence(i, f"sent {i}", 2) for i in range(20)]
    blocks = build_blocks(sents, m=20)
    assert len(blocks) == 20
    # each block has exactly 1 sentence
    for b in blocks:
        assert len(b.sentence_idxs) == 1


def test_blocks_uneven():
    sents = [Sentence(i, f"sent {i}", 2) for i in range(21)]
    blocks = build_blocks(sents, m=20)
    assert len(blocks) == 20
    total_sents = sum(len(b.sentence_idxs) for b in blocks)
    assert total_sents == 21


def test_blocks_fewer_than_m():
    sents = [Sentence(i, f"sent {i}", 2) for i in range(5)]
    blocks = build_blocks(sents, m=20)
    assert len(blocks) == 5


def test_blocks_contiguous():
    sents = [Sentence(i, f"sent {i}", 2) for i in range(40)]
    blocks = build_blocks(sents, m=20)
    all_idxs = []
    for b in blocks:
        all_idxs.extend(b.sentence_idxs)
    assert sorted(all_idxs) == list(range(40))


def test_tokens_basic():
    n = count_tokens("hello world")
    assert n > 0


def test_tokens_consistency():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)
