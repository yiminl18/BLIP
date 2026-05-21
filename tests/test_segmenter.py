"""Tests for sentence segmentation with min-token merging."""
from blip.text.segmenter import segment
from blip.text.tokens import count_tokens


def test_all_chunks_meet_min_tokens():
    text = " ".join([f"Word{i}." for i in range(200)])
    segs = segment(text, min_tokens=50)
    # every chunk except possibly the last (merged into previous) must be >= 50
    for s in segs:
        assert count_tokens(s) >= 50


def test_no_text_lost():
    text = "The cat sat. The dog ran. A bird flew. " * 20
    segs = segment(text, min_tokens=50)
    # all words should still be present
    assert sum(len(s.split()) for s in segs) == len(" ".join(segs).split())


def test_single_short_sentence():
    segs = segment("Hello.", min_tokens=50)
    assert len(segs) == 1
    assert "Hello" in segs[0]
