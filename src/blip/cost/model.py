from __future__ import annotations
from blip.llm.usage import Usage

# Price per 1M tokens (Azure gpt-4o-mini family approximation)
_C_IN_PER_M = 0.15   # $/1M input tokens
_C_OUT_PER_M = 0.60  # $/1M output tokens


def token_cost(
    usage: Usage,
    f_L: float = 2.0,
    c_in: float = _C_IN_PER_M / 1_000_000,
    c_out: float = _C_OUT_PER_M / 1_000_000,
) -> float:
    """
    Equation 2: cost with KV-cache discount.
    cached_tokens are charged at c_in / f_L.
    Uncached prompt tokens charged at c_in.
    """
    uncached = usage.prompt_tokens - usage.cached_tokens
    cost = (usage.cached_tokens * c_in / f_L) + (uncached * c_in) + (usage.completion_tokens * c_out)
    return cost


def baseline_cost(
    text_tokens: int,
    answer_tokens: int = 20,  # conservative estimate
    c_in: float = _C_IN_PER_M / 1_000_000,
    c_out: float = _C_OUT_PER_M / 1_000_000,
) -> float:
    """Cost of one uncached call over full text (Eq. 1)."""
    return text_tokens * c_in + answer_tokens * c_out
