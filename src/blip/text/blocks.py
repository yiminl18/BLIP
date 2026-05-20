from __future__ import annotations
from blip._types import Sentence, Block


def build_blocks(sentences: list[Sentence], m: int = 20) -> list[Block]:
    """Partition sentences into m equal-sized contiguous blocks."""
    n = len(sentences)
    if n == 0:
        return []
    # sizes: first (n % m) blocks get ceil(n/m) sentences, rest get floor
    base, extra = divmod(n, m)
    blocks: list[Block] = []
    start = 0
    for i in range(min(m, n)):
        size = base + (1 if i < extra else 0)
        chunk = sentences[start : start + size]
        text = " ".join(s.text for s in chunk)
        token_count = sum(s.token_count for s in chunk)
        blocks.append(Block(
            idx=i,
            sentence_idxs=tuple(s.idx for s in chunk),
            text=text,
            token_count=token_count,
        ))
        start += size
    return blocks
