"""
Precompute offline assets for the pilot sample.

Usage:
    python -m blip.runner.precompute [--n 20] [--seed 42] [--smoke-test]
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
from blip.text.tokens import count_tokens
from blip.text.blocks import build_blocks
from blip._types import Sentence
from blip.cost.accounting import CostLedger

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SAMPLES_DIR = _REPO_ROOT / "workload" / "paper" / "samples"
_RUNS_DIR = _REPO_ROOT / "runs"


def _is_answerable(answer: str) -> bool:
    return "I cannot find the answer" not in answer


def precompute(n: int = 20, seed: int = 42, dry_run: bool = False) -> None:
    cfg = load_config()
    llm = LLMClient(cfg)
    workload = PaperWorkload(seed=seed)
    ledger = CostLedger(f_L=cfg.f_L_driver)

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    run_dir = _RUNS_DIR / f"precompute_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs = workload.all_qa_pairs()
    import random
    rng = random.Random(seed)
    rng.shuffle(raw_pairs)

    sample_rows = []
    skipped = 0
    pair_id = 0

    for raw in raw_pairs:
        if len(sample_rows) >= n:
            break
        doi = raw["doi"]
        text = raw["text"]
        question = raw["question"]
        ground_truth = raw.get("ground_truth")

        if dry_run:
            logger.info("DRY RUN: would process pair %s", doi)
            continue

        try:
            answer, usage = llm.answer(text, question)
        except Exception as e:
            logger.warning("LLM call failed for %s / %s: %s", doi, question, e)
            skipped += 1
            continue

        ledger.record("precompute", usage)

        if not _is_answerable(answer):
            skipped += 1
            continue

        sents = segment(text)
        text_tokens = count_tokens(text)
        pair_id += 1
        row = {
            "pair_id": f"{pair_id:04d}",
            "doi": doi,
            "question": question,
            "ground_truth": ground_truth,
            "llm_answer": answer,
            "llm_answer_tokens": count_tokens(answer),
            "text_tokens": text_tokens,
            "text_sentence_count": len(sents),
        }
        sample_rows.append(row)
        logger.info("Pair %04d: %s tokens, answer: %s", pair_id, text_tokens, answer[:60])

    if not dry_run:
        out_file = _SAMPLES_DIR / f"sample_{n}_seed{seed}.jsonl"
        with out_file.open("w") as f:
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")
        logger.info("Wrote %d pairs to %s (skipped %d)", len(sample_rows), out_file, skipped)

        cost_summary = ledger.to_dict()
        (run_dir / "cost.json").write_text(json.dumps(cost_summary, indent=2))
        logger.info("Cost summary: %s", cost_summary)


def smoke_test(llm: LLMClient) -> None:
    """Issue one call to each deployment to verify credentials."""
    logger.info("Smoke test: driver...")
    _, u = llm.answer("Hello world.", "What is this?")
    logger.info("  driver OK: %s", u)

    logger.info("Smoke test: embedding...")
    vecs, u = llm.embed(["Hello world."])
    logger.info("  embed OK: dim=%d usage=%s", len(vecs[0]), u)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    llm = LLMClient(cfg)

    if args.smoke_test:
        smoke_test(llm)
        return

    precompute(n=args.n, seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
