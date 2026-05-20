"""Fake LLM client for unit tests. No real API calls."""
from __future__ import annotations
from blip.llm.usage import Usage


class FakeLLMClient:
    """
    Scripted fake LLM client. Callers configure answer_fn / judge_fn callbacks.
    Defaults: answer returns 'Paris', judge returns 'True'.
    """

    def __init__(
        self,
        answer_fn=None,
        judge_fn=None,
        rank_fn=None,
        provenance_fn=None,
        embed_fn=None,
    ) -> None:
        self._answer_fn = answer_fn or (lambda text, q: "Paris")
        self._judge_fn = judge_fn or (lambda a, b: "True")
        self._rank_fn = rank_fn or (lambda q, a, blocks: ",".join(["5"] * len(blocks)))
        self._provenance_fn = provenance_fn or (lambda q, a, sents: "")
        self._embed_fn = embed_fn or (lambda texts: [[0.1] * 8 for _ in texts])
        self.answer_calls: list[tuple[str, str]] = []
        self.judge_calls: list[tuple[str, str]] = []

    def _usage(self, model: str = "fake") -> Usage:
        return Usage(prompt_tokens=100, cached_tokens=0, completion_tokens=10, model=model)

    def answer(self, text: str, question: str, model: str = "driver") -> tuple[str, Usage]:
        self.answer_calls.append((text, question))
        return self._answer_fn(text, question), self._usage()

    def judge(self, a: str, b: str, model: str | None = None) -> tuple[str, Usage]:
        self.judge_calls.append((a, b))
        return self._judge_fn(a, b), self._usage()

    def rank(self, question: str, answer: str, blocks: list[str], model: str = "driver") -> tuple[str, Usage]:
        return self._rank_fn(question, answer, blocks), self._usage()

    def provenance(self, question: str, answer: str, sentences: list[str], model: str = "driver") -> tuple[str, Usage]:
        return self._provenance_fn(question, answer, sentences), self._usage()

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        return self._embed_fn(texts), self._usage()
