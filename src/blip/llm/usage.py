from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
