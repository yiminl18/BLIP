"""
Run a strategy on the pre-computed sample and write JSONL results.

Usage:
    python -m blip.runner.experiment --strategy embedding_adaptive_auto [--n 20] [--seed 42]

Strategy names: {embedding|llm}_{bottom_up|top_down|adaptive}_{none|seq|exp|auto}
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

from blip.config import load_config
from blip.llm.client import LLMClient
from blip.workloads.paper import PaperWorkload
from blip.text.segmenter import segment
from blip.text.blocks import build_blocks
from blip.text.tokens import count_tokens
from blip._types import Sentence, Pair
from blip.pipeline import StrategySpec, run
from blip.rank.embedding import EmbeddingRanker
from blip.rank.llm import LLMRanker
from blip.cache.disk import DiskCache
from blip.cost.accounting import CostLedger
from blip.cost.model import token_cost

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SAMPLES_DIR = _REPO_ROOT / "workload" / "paper" / "samples"
_RUNS_DIR = _REPO_ROOT / "runs"


def _parse_strategy(name: str) -> StrategySpec:
    """Parse strategy name like 'embedding_adaptive_auto'."""
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Strategy name must be ranker_scan_refine, got: {name}")
    ranker = parts[0]      # embedding | llm
    scan = parts[1]        # bottom_up | top_down | adaptive
    refine = parts[2]      # none | seq | exp | auto
    return StrategySpec(name=name, ranker=ranker, scan=scan, refine=refine)


def load_pairs_from_sample(sample_file: Path, workload: PaperWorkload) -> list[Pair]:
    pairs = []
    for line in sample_file.read_text().splitlines():
        row = json.loads(line)
        doi = row["doi"]
        sents_text = workload.sentences(doi)
        sents = [
            Sentence(idx=i, text=s, token_count=count_tokens(s))
            for i, s in enumerate(sents_text)
        ]
        blocks = build_blocks(sents)
        pairs.append(Pair(
            pair_id=row["pair_id"],
            doc_id=doi,
            question=row["question"],
            ground_truth=row.get("ground_truth"),
            llm_answer=row["llm_answer"],
            sentences=tuple(sents),
            blocks=tuple(blocks),
        ))
    return pairs


def run_experiment(
    strategy_name: str,
    n: int = 20,
    seed: int = 42,
) -> None:
    cfg = load_config()
    llm = LLMClient(cfg)
    cache = DiskCache(cfg.cache_dir, "embeddings")
    workload = PaperWorkload(seed=seed)

    strategy = _parse_strategy(strategy_name)

    # Build ranker
    embed_ranker = EmbeddingRanker(llm, cache=cache)
    if strategy.ranker == "llm":
        ranker = LLMRanker(llm, fallback=embed_ranker)
    else:
        ranker = embed_ranker

    sample_file = _SAMPLES_DIR / f"sample_{n}_seed{seed}.jsonl"
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample not found: {sample_file}. Run precompute first.")

    pairs = load_pairs_from_sample(sample_file, workload)

    ts = int(time.time())
    run_dir = _RUNS_DIR / f"{strategy_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "results.jsonl"
    log_file = run_dir / "log.jsonl"

    ledger = CostLedger(f_L=cfg.f_L_driver)
    verified_count = 0

    with out_file.open("w") as out_f, log_file.open("w") as log_f:
        for pair in pairs:
            try:
                result = run(
                    pair, strategy, ranker, llm,
                    refine_threshold_t=cfg.refine_threshold_t,
                    fastpath_mode=cfg.fastpath if strategy.fastpath == "off" else None,
                )
                if result.verified:
                    verified_count += 1

                row = {
                    "pair_id": result.pair_id,
                    "strategy": result.strategy,
                    "verified": result.verified,
                    "size_ratio": result.size_ratio,
                    "cost_ratio": result.cost_ratio,
                    "latency_s": result.latency_s,
                    "fastpath_hit": result.fastpath_hit,
                    "n_usages": len(result.usages),
                    "provenance_size": len(result.provenance_idxs),
                }
                out_f.write(json.dumps(row) + "\n")

                for u in result.usages:
                    log_f.write(json.dumps({
                        "pair_id": pair.pair_id,
                        "model": u.model,
                        "prompt_tokens": u.prompt_tokens,
                        "cached_tokens": u.cached_tokens,
                        "completion_tokens": u.completion_tokens,
                        "cost": token_cost(u),
                    }) + "\n")
                    ledger.record("run", u)

                logger.info(
                    "Pair %s: verified=%s size=%.3f cost=%.3fx latency=%.1fs",
                    pair.pair_id, result.verified, result.size_ratio,
                    result.cost_ratio, result.latency_s,
                )
            except Exception as e:
                logger.error("Pair %s failed: %s", pair.pair_id, e)
                out_f.write(json.dumps({"pair_id": pair.pair_id, "error": str(e)}) + "\n")

    cost_summary = ledger.to_dict()
    cost_summary["verified"] = verified_count
    cost_summary["total"] = len(pairs)
    cost_summary["accuracy"] = verified_count / len(pairs) if pairs else 0
    (run_dir / "cost.json").write_text(json.dumps(cost_summary, indent=2))
    logger.info("Done. Accuracy=%.3f, Cost=%s", cost_summary["accuracy"], cost_summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.strategy, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
