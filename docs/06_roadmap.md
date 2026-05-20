# Roadmap

Phased milestones for reproducing BLIP on the local `paper` workload. Each milestone has a concrete deliverable that can be demoed or tested independently. Estimated effort is in solo-developer-days; multiply for review and unforeseen debugging.

**Sampling default for this roadmap.** All milestones from M2 onward run on a **20-pair pilot** by default, not the full 500. Scaling to 50 or 500 pairs is conditional on the pilot looking right (verifiability == 1.0, cost ratio in the right ballpark). See `05_evaluation_plan.md` §1 and §7 for the gating logic. The roadmap below cites "the pilot sample" everywhere it used to say "500-pair sample".

## Milestone 0 — Plumbing (≈ 2–3 days)

Goal: wire up Azure, load the workload, precompute embeddings + answers for the **pilot sample (20 pairs)**. The same script can later be re-run with `--n 50` or `--n 500` once we trust the pilot results.

Deliverables:

- `src/blip/config.py` reads `local/azure.json` and exposes `Config`.
- `src/blip/llm/client.py` makes a successful chat call against the `gpt-54-mini` deployment, returns `Usage` with `cached_tokens` populated.
- `src/blip/workloads/paper.py` loads `papers.json` and exposes `iter_pairs()`.
- `src/blip/text/segmenter.py` segments document text; `blocks.py` builds `m=20` blocks; round-trip test passes.
- `src/blip/runner/precompute.py` produces `samples/sample_<n>_seed42.jsonl` with `llm_answer` cached for every pair (default `n = 20`).
- Per-doc and per-question embeddings cached (only for sampled pairs).

Exit criteria:

- Pilot sample (20 pairs) generated end-to-end; total spend < $1.
- A spot-check on 5 pairs shows `llm_answer` is the same on re-run (determinism check).
- Cost telemetry: total $ spent during precompute logged to `runs/precompute_<ts>/cost.json`.

Risks:

- Azure deployment names / region quirks. Mitigation: implement a `--smoke-test` flag that issues one call to each of the three deployments before any real run.
- `papers.json` text encoding (UTF-8 vs. Latin-1). Mitigation: explicit `encoding="utf-8"` everywhere; assert the file loads with no replacement characters.

## Milestone 1 — Judge + verification skeleton (≈ 1–2 days)

Goal: the equivalence judge works end-to-end and is calibrated.

Deliverables:

- `src/blip/llm/judge.py` implements `equivalent(a, b)` with the two-shot prompt.
- Calibration script: runs `05_evaluation_plan.md` §5 protocol on 100 pairs, reports agreement.
- A `verify(provenance, pair) -> bool` helper in `pipeline.py` used by all algorithms.

Exit criteria:

- Self-equivalence test: `equivalent(x, x)` returns True on 100 random sampled `llm_answer` strings.
- Judge agreement vs. hand-labeled 50-pair set ≥ 90%.

Risks:

- Judge flips between True and False across calls. Mitigation: log `system_fingerprint`; if it changes mid-batch, abort.

## Milestone 2 — Phase 1 (Prune) (≈ 3–4 days)

Goal: all six prune-only strategies run, verified, and metered.

Deliverables:

- `src/blip/rank/embedding.py` (cosine over cached embeddings) and `src/blip/rank/llm.py` (`LLM-Ranker-Prompt` with parser).
- `src/blip/algo/prune.py` implements `prune(scan, ranker)` per `04_algorithms.md` §1.
- `src/blip/algo/adaptive.py` implements the adaptive heuristic with `CP = 8`.
- `runner/experiment.py --strategy embedding_bottom_up` (and the other 5) writes JSONL.

Exit criteria:

- Experiment 1 from the evaluation plan runs end-to-end on the **pilot 20-pair** sample.
- 100% verification rate (else: bug).
- Cost ratio for `Embedding_adaptive` < 0.5× on the pilot. Numbers within ±20% of the paper's Qasper figures is good enough at this scale; tighter estimates wait for the 50- or 500-pair runs.

Risks:

- `LLM_ranker_prompt` returns malformed scores on some pairs. Mitigation: tolerant parser; on persistent failure, fall back to embedding ranker for that pair (logged in the JSONL row).
- Top-down binary search misses the full-T sanity check and returns nothing. Mitigation: explicit assertion at start of `prune_top_down`.

## Milestone 3 — Phase 2 (Refine) + two-phase strategies (≈ 3–4 days)

Goal: SEQ and EXP refine implementations, plus the twelve two-phase strategies.

Deliverables:

- `src/blip/algo/refine.py` implements both `sequential_greedy` and `exponential_greedy` per `04_algorithms.md` §2.
- Two-phase strategy spec registry: `EXP` if `|pruned| ≥ 10` else `SEQ` (default), plus explicit `_SEQ` and `_EXP` overrides for each.
- Experiment 2 runner.

Exit criteria:

- All twelve two-phase strategies pass 100% verification.
- Avg size ratio < 15%.
- Avg cost ratio < 1.5× of full-text answer.
- KV-cache hit rate (cached_tokens / prompt_tokens) ≥ 30% during Refine phase. If much lower, the prompt prefix is not byte-stable.

Risks:

- SEQ infinite loop if judge oscillates. Mitigation: cache judge decisions per (pair, refine pass); treat as fixed.
- Token accounting drifts from `tokens(T)` baseline. Mitigation: assert at end of each run that `verify(provenance, pair)` was called exactly once per pair and is included in the cost ledger.

## Milestone 3.5 — Fast-path wrapper (≈ 1 day)

Goal: implement the LLM-baseline front door and wire it into `pipeline.py`.

Deliverables:

- `src/blip/algo/fastpath.py` implements `elicit(pair, llm)` (LLM-Provenance-Prompt call) and the tolerant sentence-ID parser per `04_algorithms.md` §0.2.
- `pipeline.run` branches on `Config.fastpath ∈ {off, only, refine, then_blip}` per `03_architecture.md` §7.
- A small sentence-numbering helper that renders `[1] s_1\n[2] s_2\n…` for the prompt.

Exit criteria:

- Self-test: when `Config.fastpath = "off"`, the pipeline is byte-identical to the M3 baseline.
- Smoke run on 50 pairs with `Config.fastpath = "refine"`: hit rate logged; verification still 100%.
- KV-cache hit rate on the verification call ≥ 70% (it should reuse the entire system+header+text prefix from the elicitation call).

Risks:

- LLM returns sentence IDs out of range. Mitigation: parser filters to `1 ≤ id ≤ n`; on empty result, fall through to BLIP (do not silently return ∅).
- Cost regression on misses. Mitigation: report bare-BLIP and fast-path numbers side by side; defaulting to `refine` is reversible.

## Milestone 4 — Baselines + analysis (≈ 2 days)

Goal: LLM-citation baseline + RAG-1%, RAG-5%, RAG-10%; produce paper-style summary table.

Deliverables:

- `runner/experiment.py --strategy llm_citation` and `--strategy rag --p 5`.
- Aggregation script: reads all JSONL → emits `docs/results/exp_1.md`, `exp_2.md`, summary.md.
- Cost-spent summary across all experiments.

Exit criteria:

- BLIP best-of accuracy = 1.0; RAG / LLM-citation accuracy < 0.8 on average (i.e., we reproduce the headline result of paper §1).
- Total Azure spend logged and under whatever budget was set in Milestone 0 canary.

Risks:

- Embedding ranker model differs from paper (`text-embedding-3-small` vs. `all-mpnet-base-v2`). Mitigation: this is a known intentional change documented in `01_problem_definitions.md`; report numbers as "BLIP-Azure" not "BLIP".

## Milestone 5 — Optional polish (open-ended)

Pick-from menu, in priority order:

1. Adaptive crossover validation (Experiment 5). Confirms or retunes `CP` for our workload.
2. SEQ vs. EXP crossover plot (Experiment 4).
3. Top-k provenance (Algorithm 3). Adds Problem 2 support.
4. Per-question-type breakdown (paper Table 5 categories). Needs a coarse classifier.
5. Skewed-workload Experiment 3 reproduction. Requires synthetic skewing of the sample.

None of these gate "the reproduction works"; they are extensions for a deeper write-up.

## Dependency graph (Mermaid)

```mermaid
flowchart LR
    M0[M0 Plumbing] --> M1[M1 Judge]
    M0 --> M2[M2 Prune]
    M1 --> M2
    M2 --> M3[M3 Refine + Two-Phase]
    M3 --> M35[M3.5 Fast-path wrapper]
    M35 --> M4[M4 Baselines + Analysis]
    M4 --> M5[M5 Optional polish]
```

## Cross-cutting workstreams

These do not have their own milestone but run alongside every milestone:

- **Tests.** Each algorithm module ships with unit tests using a small fake LLM (in `tests/fakes/llm.py`) that returns scripted answers. No real LLM in unit tests.
- **Cost telemetry.** Every run writes `cost.json` summarizing spend by phase and model. Reviewed at the end of every milestone.
- **Logging.** Per-pair structured logs (one JSON line per LLM call) under `runs/<ts>/log.jsonl`. Used for the audit in Experiment 6 and for post-hoc debugging.

## Risks summary

| Risk                                                           | Likelihood | Impact | Mitigation                                                        |
| -------------------------------------------------------------- | ---------- | ------ | ----------------------------------------------------------------- |
| Azure `cached_tokens` not exposed → can't compute cost ratio   | Low        | High   | Check on first call (M0); fall back to manual prefix-length calc. |
| Judge agreement < 90% → minimality silently broken             | Medium     | High   | Calibration gate in M1; escalate to `gpt-54`.                     |
| LLM ranker parser fails on >5% of pairs                        | Medium     | Medium | Tolerant parser; fall back to embedding ranker per pair.          |
| SEQ refine loops on oscillating judge                          | Low        | Medium | Cache judge decisions per refine pass.                            |
| Total spend exceeds budget                                     | Medium     | Medium | 25-pair canary before each full run; downsample if needed.        |
| Provenance "verified" but spurious (correctness ≠ verifiable)  | High       | Low    | Qualitative audit in Experiment 6; report rate, do not gate.      |

## Definition of done

This reproduction is **done** when:

1. Milestones 0–4 are complete.
2. `docs/results/summary.md` contains a table matching the shape of paper Table 6 row-for-row for the strategies we implemented, computed from at least the pilot 20-pair sample.
3. The "Definition of success" in `00_overview.md` §5 is satisfied on the pilot sample (accuracy = 1.0; size/cost ratios in the right ballpark — exact numbers depend on sample size).
4. A reader can clone the repo, point `local/azure.json` at their own Azure deployment names, and re-run the pilot end-to-end for < $5.

Scaling to 50- or 500-pair samples is optional: it tightens the confidence intervals and produces numbers directly comparable to paper Table 6, but the qualitative claim "BLIP achieves accuracy 1.0 at cost ratio ~1×" is already demonstrable at 20 pairs.

Anything beyond that is Milestone 5 territory.
