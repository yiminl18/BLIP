from __future__ import annotations
import tiktoken

_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def encode(text: str) -> list[int]:
    return _enc.encode(text)
