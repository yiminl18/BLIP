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


def segment(text: str, min_tokens: int = 50) -> list[str]:
    """Split text into sentences using pysbd, then merge consecutive short sentences
    until each unit has at least min_tokens tokens."""
    from blip.text.tokens import count_tokens
    cleaned = _preprocess(text)
    raw = [s.strip() for s in _segmenter.segment(cleaned) if s.strip()]

    merged: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in raw:
        t = count_tokens(sent)
        buf.append(sent)
        buf_tokens += t
        if buf_tokens >= min_tokens:
            merged.append(" ".join(buf))
            buf = []
            buf_tokens = 0

    if buf:
        if merged:
            # append remaining short tail to the last chunk
            merged[-1] = merged[-1] + " " + " ".join(buf)
        else:
            merged.append(" ".join(buf))

    return merged
