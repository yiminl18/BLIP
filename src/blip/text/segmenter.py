from __future__ import annotations
import re
import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)

_HEADER_RE = re.compile(r"^Page \d+ of \d+\s*$", re.MULTILINE)


def _preprocess(text: str) -> str:
    text = _HEADER_RE.sub("", text)
    # normalize whitespace: collapse runs of spaces/tabs but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def segment(text: str) -> list[str]:
    """Split text into sentences using pysbd."""
    cleaned = _preprocess(text)
    sentences = _segmenter.segment(cleaned)
    # filter empty strings
    return [s.strip() for s in sentences if s.strip()]
