from __future__ import annotations
from dataclasses import dataclass, field
from blip.llm.usage import Usage
from blip.cost.model import token_cost


@dataclass
class CostLedger:
    f_L: float = 2.0
    _events: list[tuple[str, Usage]] = field(default_factory=list)

    def record(self, phase: str, usage: Usage) -> None:
        self._events.append((phase, usage))

    def total_cost(self) -> float:
        return sum(token_cost(u, f_L=self.f_L) for _, u in self._events)

    def cost_by_phase(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for phase, u in self._events:
            result[phase] = result.get(phase, 0.0) + token_cost(u, f_L=self.f_L)
        return result

    def total_tokens(self) -> dict[str, int]:
        pt = sum(u.prompt_tokens for _, u in self._events)
        ct = sum(u.cached_tokens for _, u in self._events)
        comp = sum(u.completion_tokens for _, u in self._events)
        return {"prompt": pt, "cached": ct, "completion": comp}

    def cost_ratio(self, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return self.total_cost() / baseline

    def to_dict(self) -> dict:
        return {
            "total_cost": self.total_cost(),
            "by_phase": self.cost_by_phase(),
            "tokens": self.total_tokens(),
            "n_calls": len(self._events),
        }
