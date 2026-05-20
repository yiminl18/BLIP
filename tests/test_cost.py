"""Tests for cost model and ledger."""
from blip.llm.usage import Usage
from blip.cost.model import token_cost, baseline_cost
from blip.cost.accounting import CostLedger


def test_token_cost_no_cache():
    u = Usage(prompt_tokens=1000, cached_tokens=0, completion_tokens=100, model="x")
    cost = token_cost(u, f_L=2.0)
    expected = 1000 * (0.15 / 1_000_000) + 100 * (0.60 / 1_000_000)
    assert abs(cost - expected) < 1e-12


def test_token_cost_with_cache():
    u = Usage(prompt_tokens=1000, cached_tokens=500, completion_tokens=100, model="x")
    cost = token_cost(u, f_L=2.0)
    c_in = 0.15 / 1_000_000
    c_out = 0.60 / 1_000_000
    expected = (500 * c_in / 2.0) + (500 * c_in) + (100 * c_out)
    assert abs(cost - expected) < 1e-12


def test_ledger_accumulates():
    ledger = CostLedger(f_L=2.0)
    u1 = Usage(100, 0, 10, "x")
    u2 = Usage(200, 100, 20, "x")
    ledger.record("prune", u1)
    ledger.record("refine", u2)
    assert ledger.total_cost() == token_cost(u1) + token_cost(u2)


def test_ledger_cost_ratio():
    ledger = CostLedger(f_L=2.0)
    u = Usage(100, 0, 10, "x")
    ledger.record("prune", u)
    baseline = token_cost(u) * 2
    assert abs(ledger.cost_ratio(baseline) - 0.5) < 1e-10


def test_ledger_by_phase():
    ledger = CostLedger()
    ledger.record("prune", Usage(100, 0, 10, "x"))
    ledger.record("refine", Usage(50, 0, 5, "x"))
    by_phase = ledger.cost_by_phase()
    assert "prune" in by_phase
    assert "refine" in by_phase
    assert by_phase["prune"] > by_phase["refine"]
