"""Tests for the equivalence judge."""
import pytest
from blip.llm.judge import equivalent, _normalize, _is_closed_domain
from tests.fakes.llm import FakeLLMClient


def test_closed_domain_yes():
    equiv, _ = equivalent("yes", "yes", closed_domain=True)
    assert equiv is True


def test_closed_domain_no_mismatch():
    equiv, _ = equivalent("yes", "no", closed_domain=True)
    assert equiv is False


def test_closed_domain_year():
    equiv, _ = equivalent("2019", "2019", closed_domain=True)
    assert equiv is True


def test_closed_domain_year_mismatch():
    equiv, _ = equivalent("2018", "2019", closed_domain=True)
    assert equiv is False


def test_llm_judge_true():
    llm = FakeLLMClient(judge_fn=lambda a, b: "True")
    equiv, usages = equivalent("cats sleep", "felines rest", llm_client=llm)
    assert equiv is True
    assert len(usages) == 1


def test_llm_judge_false():
    llm = FakeLLMClient(judge_fn=lambda a, b: "False")
    equiv, usages = equivalent("profit", "loss", llm_client=llm)
    assert equiv is False


def test_llm_judge_parse_failure_retries():
    calls = []
    def bad_judge(a, b):
        calls.append(1)
        return "maybe"  # unparseable
    llm = FakeLLMClient(judge_fn=bad_judge)
    equiv, usages = equivalent("a", "b", llm_client=llm)
    assert len(calls) == 2  # retried once
    assert equiv is False  # defaulted to False


def test_is_closed_domain_yes():
    assert _is_closed_domain("Yes") is True


def test_is_closed_domain_year():
    assert _is_closed_domain("2023") is True


def test_is_closed_domain_long():
    assert _is_closed_domain("a very long answer that should not be closed domain") is False


def test_normalize():
    assert _normalize("  Hello, World!  ") == "hello world"
