from __future__ import annotations
import numpy as np
from typing import Sequence

from blip._types import Block, Pair
from blip.llm.client import LLMClient
from blip.cache.disk import DiskCache


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingRanker:
    name = "embedding"

    def __init__(self, llm: LLMClient, cache: DiskCache | None = None) -> None:
        self._llm = llm
        self._cache = cache

    def _embed(self, text: str) -> np.ndarray:
        key = f"embed:{text}"
        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return np.array(cached)
        vecs, _ = self._llm.embed([text])
        vec = np.array(vecs[0])
        if self._cache:
            self._cache.set(key, vec.tolist())
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        results = [None] * len(texts)
        to_fetch_idxs = []
        to_fetch_texts = []
        for i, t in enumerate(texts):
            key = f"embed:{t}"
            if self._cache:
                cached = self._cache.get(key)
                if cached is not None:
                    results[i] = np.array(cached)
                    continue
            to_fetch_idxs.append(i)
            to_fetch_texts.append(t)

        if to_fetch_texts:
            vecs, _ = self._llm.embed(to_fetch_texts)
            for i, idx in enumerate(to_fetch_idxs):
                vec = np.array(vecs[i])
                results[idx] = vec
                if self._cache:
                    key = f"embed:{to_fetch_texts[i]}"
                    self._cache.set(key, vec.tolist())

        return results

    def rank(self, items: Sequence[Block], pair: Pair) -> list[int]:
        query = f"{pair.question} {pair.llm_answer}"
        query_vec = self._embed(query)
        block_texts = [b.text for b in items]
        block_vecs = self.embed_batch(block_texts)
        scores = [_cosine(query_vec, bv) for bv in block_vecs]
        # sort by descending score
        return sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
