from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str
    token_count: int


@dataclass(frozen=True)
class Block:
    idx: int
    sentence_idxs: tuple[int, ...]
    text: str
    token_count: int


@dataclass(frozen=True)
class Pair:
    pair_id: str
    doc_id: str
    question: str
    ground_truth: str | None
    llm_answer: str
    sentences: tuple[Sentence, ...]
    blocks: tuple[Block, ...]


@dataclass
class Usage:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    model: str


@dataclass
class ProvenanceResult:
    pair_id: str
    strategy: str
    provenance_idxs: tuple[int, ...]
    size_ratio: float
    cost_ratio: float
    latency_s: float
    usages: list[Usage]
    verified: bool
    fastpath_hit: bool = False
