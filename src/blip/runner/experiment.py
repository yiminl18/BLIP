"""
Run a strategy on the pre-computed sample and write JSONL results.

Usage:
    python -m blip.runner.experiment --strategy embedding_adaptive_exp [--n 20] [--seed 42]

Strategy names: {embedding|llm}_{bottom_up|top_down|adaptive}_{none|seq|exp|auto}[_{off|only|refine|then_blip}]

The fastpath component is optional; it defaults to "refine" (LLM-citation fast path).
To disable: embedding_adaptive_exp_off
"""
from __future__ import annotations
import argparse
import datetime
import json
import logging
import subprocess
import time
from pathlib import Path

from blip.config import load_config
from blip.llm.client import LLMClient
from blip.llm.judge import equivalent
from blip.workloads.paper import PaperWorkload
from blip.text.blocks import build_blocks
from blip.text.tokens import count_tokens
from blip._types import Sentence, Pair, ProvenanceResult
from blip.llm.usage import Usage
from blip.pipeline import StrategySpec, run
from blip.algo.prune import SkipPair
from blip.rank.embedding import EmbeddingRanker
from blip.rank.llm import LLMRanker
from blip.cache.disk import DiskCache
from blip.cost.accounting import CostLedger
from blip.cost.model import token_cost

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SAMPLES_DIR = _REPO_ROOT / "workload" / "paper" / "samples"
_RUNS_DIR = _REPO_ROOT / "runs"

_FASTPATH_VALUES = {"off", "only", "refine", "then_blip"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _parse_strategy(name: str) -> StrategySpec:
    """Parse strategy name like 'embedding_adaptive_exp' or 'embedding_adaptive_exp_off'.

    Format: {ranker}_{scan}_{refine}[_{fastpath}]
    fastpath defaults to "refine" (LLM-citation fast path always on).
    """
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Strategy name must be ranker_scan_refine[_fastpath], got: {name}")
    ranker = parts[0]   # embedding | llm
    scan = parts[1]     # bottom_up | top_down | adaptive
    refine = parts[2]   # none | seq | exp | auto
    fastpath = parts[3] if len(parts) >= 4 and parts[3] in _FASTPATH_VALUES else "refine"
    return StrategySpec(name=name, ranker=ranker, scan=scan, refine=refine, fastpath=fastpath)


def load_pairs_from_sample(sample_file: Path, workload: PaperWorkload) -> list[tuple[Pair, dict]]:
    """Return (Pair, raw_row) tuples so the runner can access raw fields."""
    result = []
    for line in sample_file.read_text().splitlines():
        row = json.loads(line)
        doi = row["doi"]
        sents_text = workload.sentences(doi)
        sents = [
            Sentence(idx=i, text=s, token_count=count_tokens(s))
            for i, s in enumerate(sents_text)
        ]
        blocks = build_blocks(sents)
        pair = Pair(
            pair_id=row["pair_id"],
            doc_id=doi,
            question=row["question"],
            ground_truth=row.get("ground_truth"),
            llm_answer=row["llm_answer"],
            sentences=tuple(sents),
            blocks=tuple(blocks),
        )
        result.append((pair, row))
    return result


def _phase_token_summary(phase_usages: list[tuple[str, Usage]]) -> dict:
    by_phase: dict[str, dict] = {}
    for phase, u in phase_usages:
        if phase not in by_phase:
            by_phase[phase] = {"in": 0, "in_cached": 0, "out": 0}
        by_phase[phase]["in"] += u.prompt_tokens
        by_phase[phase]["in_cached"] += u.cached_tokens
        by_phase[phase]["out"] += u.completion_tokens
    return by_phase


def _phase_cost_summary(phase_usages: list[tuple[str, Usage]]) -> dict:
    by_phase: dict[str, float] = {}
    for phase, u in phase_usages:
        by_phase[phase] = by_phase.get(phase, 0.0) + token_cost(u)
    return by_phase


def _phase_call_counts(phase_usages: list[tuple[str, Usage]]) -> dict:
    counts: dict[str, int] = {}
    for phase, _ in phase_usages:
        counts[phase] = counts.get(phase, 0) + 1
    counts["total"] = len(phase_usages)
    return counts


def _judge_gt(answer: str, ground_truth: str | None, llm: LLMClient, pair: Pair) -> bool | None:
    """Compare answer to ground truth using the judge. Returns None if no ground truth."""
    if ground_truth is None:
        return None
    is_eq, _ = equivalent(answer, ground_truth, llm_client=llm, pair=pair)
    return is_eq


def _build_row(
    result: ProvenanceResult,
    pair: Pair,
    raw_row: dict,
    cfg,
    git_sha: str,
    run_id: str,
    llm: LLMClient,
) -> dict:
    sent_map = {s.idx: s.text for s in pair.sentences}
    provenance_sentences = [sent_map[i] for i in result.provenance_idxs]
    provenance_tokens = sum(
        s.token_count for s in pair.sentences if s.idx in set(result.provenance_idxs)
    )
    text_tokens = raw_row.get("text_tokens", sum(s.token_count for s in pair.sentences))
    text_sentences = raw_row.get("text_sentence_count", len(pair.sentences))

    total_cost_usd = sum(token_cost(u) for u in result.usages)
    cache_hit_rate = (
        sum(u.cached_tokens for u in result.usages) /
        max(sum(u.prompt_tokens for u in result.usages), 1)
    )

    acc_baseline_vs_gt = _judge_gt(pair.llm_answer, pair.ground_truth, llm, pair)
    acc_answer_vs_gt = _judge_gt(result.final_answer, pair.ground_truth, llm, pair)

    return {
        # --- identification ---
        "pair_id": result.pair_id,
        "doc_id": pair.doc_id,
        "question": pair.question,
        "strategy": result.strategy,
        "fastpath_mode": result.strategy.split("_")[3] if len(result.strategy.split("_")) > 3 else "refine",
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",

        # --- inputs ---
        "text_tokens": text_tokens,
        "text_sentences": text_sentences,
        "baseline_answer": pair.llm_answer,
        "baseline_answer_tokens": count_tokens(pair.llm_answer),
        "ground_truth": pair.ground_truth,

        # --- provenance output ---
        "provenance_idxs": list(result.provenance_idxs),
        "provenance_sentences": provenance_sentences,
        "provenance_size": len(result.provenance_idxs),
        "provenance_tokens": provenance_tokens,
        "size_ratio": result.size_ratio,

        # --- accuracy ---
        "accuracy": {
            "accuracy_provenance": result.verified,
            "accuracy_answer_vs_gt": acc_answer_vs_gt,
            "accuracy_baseline_vs_gt": acc_baseline_vs_gt,
        },
        "verified": result.verified,
        "final_answer": result.final_answer,
        "final_answer_tokens": count_tokens(result.final_answer),

        # --- fast-path ---
        "fastpath_hit": result.fastpath_hit,

        # --- llm call counts by phase ---
        "llm_calls": _phase_call_counts(result.phase_usages),

        # --- token totals ---
        "tokens": {
            "input_total": sum(u.prompt_tokens for u in result.usages),
            "input_cached_total": sum(u.cached_tokens for u in result.usages),
            "output_total": sum(u.completion_tokens for u in result.usages),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "by_phase": _phase_token_summary(result.phase_usages),
        },

        # --- cost ---
        "cost_usd": {
            "baseline": result.baseline_cost_usd,
            "total": total_cost_usd,
            "by_phase": _phase_cost_summary(result.phase_usages),
        },
        "cost_ratio": result.cost_ratio,

        # --- latency ---
        "latency_s": result.latency_s,

        # --- config snapshot ---
        "config": {
            "model_driver": cfg.azure.driver.deployment,
            "model_judge": cfg.azure.driver.deployment,
            "model_embed": cfg.azure.embed.deployment,
            "block_count_m": cfg.block_count_m,
            "refine_threshold_t": cfg.refine_threshold_t,
            "f_L_driver": cfg.f_L_driver,
            "judge_prompt": cfg.judge_prompt,
            "seed": cfg.seed,
        },
    }


def run_experiment(
    strategy_name: str,
    n: int = 20,
    seed: int = 42,
    max_docs: int | None = None,
) -> None:
    cfg = load_config()
    llm = LLMClient(cfg)
    cache = DiskCache(cfg.cache_dir, "embeddings")
    workload = PaperWorkload(seed=seed)
    git_sha = _git_sha()

    strategy = _parse_strategy(strategy_name)

    embed_ranker = EmbeddingRanker(llm, cache=cache)
    if strategy.ranker == "llm":
        ranker = LLMRanker(llm, fallback=embed_ranker)
    else:
        ranker = embed_ranker

    doc_tag = f"_docs{max_docs}" if max_docs is not None else ""
    sample_file = _SAMPLES_DIR / f"sample_{n}_seed{seed}{doc_tag}.jsonl"
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample not found: {sample_file}. Run precompute first.")

    pairs_with_rows = load_pairs_from_sample(sample_file, workload)

    ts = int(time.time())
    run_id = f"{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M')}-{ts % 10000:04x}"
    run_dir = _RUNS_DIR / f"{strategy_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "results.jsonl"
    log_file = run_dir / "log.jsonl"

    ledger = CostLedger(f_L=cfg.f_L_driver)
    success_count = 0
    skip_count = 0

    with out_file.open("w") as out_f, log_file.open("w") as log_f:
        for pair, raw_row in pairs_with_rows:
            try:
                result = run(
                    pair, strategy, ranker, llm,
                    refine_threshold_t=cfg.refine_threshold_t,
                )
                success_count += 1

                row = _build_row(result, pair, raw_row, cfg, git_sha, run_id, llm)
                out_f.write(json.dumps(row) + "\n")

                for phase, u in result.phase_usages:
                    log_f.write(json.dumps({
                        "pair_id": pair.pair_id,
                        "phase": phase,
                        "model": u.model,
                        "prompt_tokens": u.prompt_tokens,
                        "cached_tokens": u.cached_tokens,
                        "completion_tokens": u.completion_tokens,
                        "cost": token_cost(u),
                    }) + "\n")
                    ledger.record(phase, u)

                logger.info(
                    "Pair %s: size=%.3f cost=%.3fx latency=%.1fs fp_hit=%s",
                    pair.pair_id, result.size_ratio,
                    result.cost_ratio, result.latency_s, result.fastpath_hit,
                )
            except SkipPair as e:
                skip_count += 1
                logger.info("Pair %s skipped: %s", pair.pair_id, e)
                out_f.write(json.dumps({"pair_id": pair.pair_id, "skipped": True, "reason": str(e)}) + "\n")
            except Exception as e:
                logger.error("Pair %s failed: %s", pair.pair_id, e, exc_info=True)
                out_f.write(json.dumps({"pair_id": pair.pair_id, "error": str(e)}) + "\n")

    cost_summary = ledger.to_dict()
    cost_summary["success"] = success_count
    cost_summary["skipped"] = skip_count
    cost_summary["total"] = len(pairs_with_rows)
    cost_summary["run_id"] = run_id
    cost_summary["git_sha"] = git_sha
    (run_dir / "cost.json").write_text(json.dumps(cost_summary, indent=2))
    logger.info("Done. success=%d skipped=%d total_cost=$%.4f", success_count, skip_count, cost_summary["total_cost"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.strategy, n=args.n, seed=args.seed, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
