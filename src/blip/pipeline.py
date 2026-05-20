from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Literal

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
from blip.text.tokens import count_tokens


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


def run(
    pair: Pair,
    strategy: StrategySpec,
    ranker: Ranker,
    llm: LLMClient,
    refine_threshold_t: int = 10,
    fastpath_mode: str | None = None,
) -> ProvenanceResult:
    t0 = time.perf_counter()
    all_usages: list[Usage] = []
    fastpath_hit = False
    fp_mode = fastpath_mode if fastpath_mode is not None else strategy.fastpath

    # 0. Optional fast-path
    if fp_mode != "off":
        fp_idxs = fastpath_mod.elicit(pair, llm)
        if fp_idxs:
            is_ver, fp_usages = fastpath_mod.verify_provenance(fp_idxs, pair, llm)
            all_usages.extend(fp_usages)
            if is_ver:
                fastpath_hit = True
                if fp_mode == "refine":
                    fp_idxs, ref_usages = _do_refine(fp_idxs, pair, llm, "auto", refine_threshold_t)
                    all_usages.extend(ref_usages)
                latency = time.perf_counter() - t0
                return ProvenanceResult(
                    pair_id=pair.pair_id,
                    strategy=strategy.name,
                    provenance_idxs=tuple(sorted(fp_idxs)),
                    size_ratio=_size_ratio(fp_idxs, pair),
                    cost_ratio=_cost_ratio(all_usages, pair),
                    latency_s=latency,
                    usages=all_usages,
                    verified=True,
                    fastpath_hit=True,
                )
        if fp_mode == "only":
            # fast-path failed and mode is "only" → still need to return something
            # fall through to BLIP but mark as miss
            pass

    # 1. Prune
    if strategy.scan == "adaptive":
        pruned_idxs, prune_usages = adaptive_mod.adaptive_prune(pair, ranker, llm)
    else:
        pruned_idxs, prune_usages = prune_mod.prune(pair, ranker, llm, scan=strategy.scan)
    all_usages.extend(prune_usages)

    # 2. Refine
    refined_idxs, ref_usages = _do_refine(
        pruned_idxs, pair, llm, strategy.refine, refine_threshold_t
    )
    all_usages.extend(ref_usages)

    # Final verification (by construction this must hold; assert catches bugs)
    text = _text_of(refined_idxs, pair)
    answer, verify_usage = llm.answer(text, pair.question)
    all_usages.append(verify_usage)
    is_ver, judge_usages = equivalent(answer, pair.llm_answer, llm_client=llm, pair=pair)
    all_usages.extend(judge_usages)
    assert is_ver, f"BUG: returned provenance does not verify on pair {pair.pair_id}"

    latency = time.perf_counter() - t0
    return ProvenanceResult(
        pair_id=pair.pair_id,
        strategy=strategy.name,
        provenance_idxs=tuple(sorted(refined_idxs)),
        size_ratio=_size_ratio(refined_idxs, pair),
        cost_ratio=_cost_ratio(all_usages, pair),
        latency_s=latency,
        usages=all_usages,
        verified=True,
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


def _cost_ratio(usages: list[Usage], pair: Pair) -> float:
    total_cost = sum(token_cost(u) for u in usages)
    text_tokens = sum(s.token_count for s in pair.sentences)
    base = baseline_cost(text_tokens)
    if base == 0:
        return 0.0
    return total_cost / base
