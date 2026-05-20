from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Literal

from blip._types import Pair, ProvenanceResult
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent
from blip.llm.usage import Usage
from blip.rank.base import Ranker
from blip.algo import prune as prune_mod
from blip.algo import refine as refine_mod
from blip.algo import adaptive as adaptive_mod
from blip.algo import fastpath as fastpath_mod
from blip.cost.model import baseline_cost, token_cost


@dataclass
class StrategySpec:
    name: str
    ranker: str = "embedding"        # "embedding" | "llm"
    scan: str = "bottom_up"          # "bottom_up" | "top_down" | "adaptive"
    refine: str = "auto"             # "none" | "seq" | "exp" | "auto"
    fastpath: str = "refine"         # "off" | "only" | "refine" | "then_blip"


def _text_of(sentence_idxs: list[int], pair: Pair) -> str:
    sent_map = {s.idx: s.text for s in pair.sentences}
    return " ".join(sent_map[i] for i in sorted(sentence_idxs))


def _size_ratio(sentence_idxs: list[int], pair: Pair) -> float:
    p_tokens = sum(
        s.token_count for s in pair.sentences if s.idx in set(sentence_idxs)
    )
    t_tokens = sum(s.token_count for s in pair.sentences)
    if t_tokens == 0:
        return 1.0
    return p_tokens / t_tokens


def _base_cost(pair: Pair) -> float:
    text_tokens = sum(s.token_count for s in pair.sentences)
    return baseline_cost(text_tokens)


def _cost_ratio(phase_usages: list[tuple[str, Usage]], pair: Pair) -> float:
    total = sum(token_cost(u) for _, u in phase_usages)
    base = _base_cost(pair)
    return total / base if base else 0.0


def run(
    pair: Pair,
    strategy: StrategySpec,
    ranker: Ranker,
    llm: LLMClient,
    refine_threshold_t: int = 10,
    fastpath_mode: str | None = None,
) -> ProvenanceResult:
    t0 = time.perf_counter()
    phase_usages: list[tuple[str, Usage]] = []
    fastpath_hit = False
    fp_mode = fastpath_mode if fastpath_mode is not None else strategy.fastpath

    def _record(phase: str, usages: list[Usage]) -> None:
        for u in usages:
            phase_usages.append((phase, u))

    # 0. Optional fast-path
    if fp_mode != "off":
        fp_idxs = fastpath_mod.elicit(pair, llm)
        if fp_idxs:
            is_ver, fp_usages = fastpath_mod.verify_provenance(fp_idxs, pair, llm)
            _record("fastpath_verify", fp_usages)
            if is_ver:
                fastpath_hit = True
                if fp_mode == "refine":
                    fp_idxs, ref_usages = _do_refine(fp_idxs, pair, llm, "auto", refine_threshold_t)
                    _record("refine", ref_usages)
                latency = time.perf_counter() - t0
                final_answer = _get_final_answer(fp_idxs, pair, llm, phase_usages)
                return _make_result(
                    pair, strategy, fp_idxs, phase_usages,
                    final_answer, latency, verified=True, fastpath_hit=True,
                )

    # 1. Prune
    if strategy.scan == "adaptive":
        pruned_idxs, prune_usages = adaptive_mod.adaptive_prune(pair, ranker, llm)
    else:
        pruned_idxs, prune_usages = prune_mod.prune(pair, ranker, llm, scan=strategy.scan)
    _record("prune", prune_usages)

    # 2. Refine
    refined_idxs, ref_usages = _do_refine(
        pruned_idxs, pair, llm, strategy.refine, refine_threshold_t
    )
    _record("refine", ref_usages)

    # Final verification
    text = _text_of(refined_idxs, pair)
    answer, verify_usage = llm.answer(text, pair.question)
    _record("verify_final", [verify_usage])
    is_ver, judge_usages = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
    _record("judge", judge_usages)
    assert is_ver, f"BUG: returned provenance does not verify on pair {pair.pair_id}"

    latency = time.perf_counter() - t0
    return _make_result(
        pair, strategy, refined_idxs, phase_usages,
        answer, latency, verified=True, fastpath_hit=False,
    )


def _get_final_answer(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
    phase_usages: list[tuple[str, Usage]],
) -> str:
    text = _text_of(sentence_idxs, pair)
    answer, u = llm.answer(text, pair.question)
    phase_usages.append(("verify_final", u))
    return answer


def _make_result(
    pair: Pair,
    strategy: StrategySpec,
    sentence_idxs: list[int],
    phase_usages: list[tuple[str, Usage]],
    final_answer: str,
    latency: float,
    verified: bool,
    fastpath_hit: bool,
) -> ProvenanceResult:
    flat_usages = [u for _, u in phase_usages]
    return ProvenanceResult(
        pair_id=pair.pair_id,
        strategy=strategy.name,
        provenance_idxs=tuple(sorted(sentence_idxs)),
        size_ratio=_size_ratio(sentence_idxs, pair),
        cost_ratio=_cost_ratio(phase_usages, pair),
        latency_s=latency,
        usages=flat_usages,
        phase_usages=phase_usages,
        verified=verified,
        final_answer=final_answer,
        baseline_cost_usd=_base_cost(pair),
        fastpath_hit=fastpath_hit,
    )


def _do_refine(
    sentence_idxs: list[int],
    pair: Pair,
    llm: LLMClient,
    refine_mode: str,
    threshold_t: int,
) -> tuple[list[int], list[Usage]]:
    if refine_mode == "none":
        return sentence_idxs, []
    if refine_mode == "seq" or (refine_mode == "auto" and len(sentence_idxs) < threshold_t):
        return refine_mod.sequential_greedy(sentence_idxs, pair, llm)
    else:
        return refine_mod.exponential_greedy(sentence_idxs, pair, llm)
