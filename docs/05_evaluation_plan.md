# Evaluation Plan

The reproduction succeeds when, on the local `paper` workload with the `gpt-54-mini` Azure deployment, BLIP returns provenance that satisfies the verifiability guarantee at a cost ratio comparable to the paper's Qasper numbers. This document specifies metrics, experiments, baselines, and the judging protocol.

## 1. Sample under test

**Default protocol is small-sample-first.** Burning the full 500-pair sample × 18 strategies × every fast-path variant up front is wasteful. The plan is:

1. **Pilot — 20 pairs.** Run every strategy on 20 fixed pairs (seed `42`, first 20 from the shuffled order). This is the unit of work for "is this strategy working at all?" Total Azure spend: $0.50–$2 across all strategies. Used to debug, calibrate the judge, and confirm verifiability == 1.0.
2. **Small — 50 pairs (optional).** Re-run the *handful of strategies that survive the pilot* (typically `Embedding_adaptive_EXP`, `LLM_bottom_up_EXP`, and the two fast-path variants). Used to tighten the mean estimates before any final decision. Spend: a few dollars.
3. **Full — 500 pairs (only if results look right at 50).** Same as the paper's protocol. Spend: $10–$30 depending on strategies enabled. Treated as the *final* run, not the *first* run.

Steps 2 and 3 are conditional. **It is fine, and the expected outcome for an initial reproduction**, to stop at step 1 or 2. The paper-comparable headline numbers come from step 3, but the qualitative claim "BLIP achieves accuracy 1.0 with cost ratio ~1×" can be made from 20–50 pairs.

`A = L(T, Q)` is computed once per pair (during sample precompute) and stored in `cache/answers/`. **All strategies use this exact same `A`** as the verification target. This is non-negotiable: comparing strategies against different `A`'s makes the comparison meaningless.

Sample-size selection note: standard error on a verifiability rate of ~1.0 is ≤ 0.05 at `n = 20` (Wilson interval, 95% CI). For accuracy near 1.0, small samples are *more* informative than for accuracy near 0.5, so 20 pairs is a real test, not just a smoke test.

## 1.5 Strategy catalog

The implementation surface is three orthogonal axes. Their product is large; the *pilot run* uses a focused subset (§1.6).

**Axis 1 — Pruning (Phase 1).** Six options = ranker {Embedding, LLM} × scan {bottom_up, top_down, adaptive}.

**Axis 2 — Refinement (Phase 2).** Three options:
- `none` (prune-only; provenance may not be minimal)
- `SEQ` (`Sequential_Greedy`)
- `EXP` (`Exponential_Greedy`)

→ 6 × 3 = **18 BLIP combinations** (`Embedding_adaptive_EXP`, `LLM_bottom_up_SEQ`, …).

**Axis 3 — Fast-path mode.** Four wrapper modes, orthogonal to the 18:
- `off` (bare BLIP)
- `only` (return LLM-citation as-is on hit; gives up minimality)
- `refine` (run Refine on LLM-citation on hit; preserves minimality) — **default**
- `then_blip` (alias of `only` for the result label)

→ 18 × 4 = **72 total combinations** in the strategy spec surface.

**Optional, feature-flagged.**
- `top_k` (Algorithm 3) — wraps any BLIP combination, returns up to `k` distinct minimal provenances. Milestone-2.

**Baselines (not BLIP).**
- `LLM_citation` — one `LLM-Provenance-Prompt` call, return cited sentences, no verification gate. Paper §1 / Table 1.
- `RAG_p` for `p ∈ {1, 5, 10}` — retrieve top-`p%` sentences by cosine to `Q + A`. Paper §1 / Table 1.

## 1.6 Pilot subset (what we actually run first)

Seven strategies × 20 pairs ≈ 140 strategy-runs, total spend < $2:

| #  | Strategy                                           | Why include it                                                      |
| -- | -------------------------------------------------- | ------------------------------------------------------------------- |
| 1  | `Embedding_adaptive_EXP`                           | Paper's recommended adaptive default — the BLIP headline.           |
| 2  | `Embedding_adaptive_EXP` × `fastpath=refine`       | Fast-path-augmented headline; measures hit rate and savings.        |
| 3  | `LLM_bottom_up_EXP`                                | Top LLM-ranker strategy on Qasper (paper Table 6 bold).             |
| 4  | `Embedding_bottom_up_EXP`                          | Top embedding-ranker strategy on Qasper (paper Table 6 bold).       |
| 5  | `Embedding_adaptive` (prune only, no refine)       | Cheapest BLIP — cost-vs-minimality reference.                       |
| 6  | `LLM_citation`                                     | Baseline. Paper Table 1: ~65% accuracy on gpt-4o-mini.              |
| 7  | `RAG_5`                                            | Baseline. Paper Table 1: ~46% accuracy on Qasper.                   |

The remaining 65 BLIP combinations and the other two RAG sizes are reachable by changing one config flag. They exist as ablations to run only if a specific question (e.g., "does EXP help when the provenance is small?") comes up.

## 2. Metrics

| Metric            | Definition                                                                                                  | Paper ref           |
| ----------------- | ----------------------------------------------------------------------------------------------------------- | ------------------- |
| Accuracy          | Fraction of pairs where the returned `P` satisfies `I(L(P, Q), A) = True`. **Should be 1.0 by construction.** | §1, §5.2 Exp. 2     |
| Size ratio        | `tokens(P) / tokens(T)`, averaged across pairs.                                                             | §5.2 Exp. 1         |
| Cost ratio        | `cost(strategy) / cost(answer(Q, T))`, averaged across pairs.                                               | §5.2 Exp. 1         |
| Latency           | Wall-clock seconds per pair, averaged.                                                                      | §5.2 Exp. 1         |
| Human review (HR) | One-pass estimate: `size_ratio` if verified, else `1.0`.                                                    | §5.2 Exp. 2 / Tbl 7 |

Recovery metrics `R` and `eR` are skipped — no ground-truth provenance tuples in this workload.

Strategies under test include both bare BLIP variants and BLIP-with-fastpath variants. The fast-path mode (`off | only | refine | then_blip`) is a strategy-spec field; results are reported separately per mode (see Experiment 7).

Cost computation:
- Numerator = sum of `cost(Usage)` for every LLM call the strategy made, including the final verification call. Embedding API calls are excluded (treated as offline, per paper convention).
- Denominator = `cost(answer(Q, T))` for that pair, computed at `temperature=0` with empty cache. Pre-computed during the sample-generation step.

## 3. Experiments

### Experiment 1 — Pruning strategies (mirror paper Table 6 / 10, prune block)

Six prune-only strategies, no Refine:

1. `Embedding_bottom_up`
2. `Embedding_top_down`
3. `Embedding_adaptive`
4. `LLM_bottom_up`
5. `LLM_top_down`
6. `LLM_adaptive`

Report: size ratio, cost ratio, latency. Expected pattern (from paper Qasper rows): adaptive beats both pure scans on cost; embedding rankers are cheaper than LLM rankers but slightly larger provenance.

### Experiment 2 — Two-phase strategies (mirror paper Table 6, two-phase block)

Twelve two-phase strategies = {6 prune} × {SEQ, EXP}:

`{Embedding,LLM}_{bottom_up,top_down,adaptive}_{SEQ,EXP}`

Report: size ratio, cost ratio, latency. Expected pattern: size ratio drops to single-digit %, cost ratio rises to ~1× of full-text. The bolded "top-3" strategies in the paper's Table 6 (Qasper column) are:

- `LLM_bottom_up_EXP`
- `Embedding_bottom_up_EXP`
- `Embedding_adaptive_EXP`

We expect a similar top-3 on our workload (modulo ranking by question type — the `paper` corpus is more lookup-heavy than Qasper).

### Experiment 3 — Bottom-up vs. top-down vs. adaptive (mirror paper Fig. 8, Table 9)

For both rankers separately, plot cost ratio for `bottom_up`, `top_down`, `adaptive` across the sample. Report:

- Crossover `CP_1` (block count at which BU vs. TD cross). Compare to the theoretical `CP ≈ 8`.
- Q1 / Q3 of returned block counts. If most pairs return ≤ `CP` blocks, adaptive should track BU; otherwise it should track TD.

### Experiment 4 — SEQ vs. EXP (mirror paper Fig. 10, Table 9 right cols)

Fix prune = `LLM_bottom_up`. Run both `SEQ` and `EXP` refine on the same pruned outputs. Plot cost ratio vs. `|pruned|`. Expect a crossover around 10 sentences (paper's recommendation).

### Experiment 5 — Adaptive crossover validation

For each pair, record (`returned_block_count`, `bu_won`, `td_won`). Compute the empirical crossover point and check it against `CP ≈ 8`. If the empirical CP is far off (>4 blocks), retune the formula for our workload.

### Experiment 6 — Baselines (mirror paper Table 1 + Table 7 RAG rows)

- **LLM-citation baseline.** One call per pair using `LLM-Provenance-Prompt` (paper §1). Parse returned sentence IDs, build `P`, verify with `I(L(P, Q), A)`. Report accuracy and avg size ratio.
- **RAG baselines.** Retrieval over sentence embeddings, return the top-`p%` sentences by cosine to `Q + A`. Run with `p ∈ {1, 5, 10}`. Report accuracy and avg size ratio.

These two baselines are the comparison columns BLIP must beat on accuracy. Paper expectation: RAG-10% ~ 0.7–0.78 accuracy on Qasper; LLM-citation ~ 0.65–0.74 (Table 1). BLIP should be 1.0.

### Experiment 7 — Fast-path hit rate and end-to-end savings

The LLM-baseline fast-path (algorithms doc §0) is a configurable front door. This experiment quantifies its impact.

For each of the four fast-path settings (`off`, `only`, `refine`, `then_blip`), running the recommended strategy `Embedding_adaptive_EXP` underneath:

- **Hit rate.** Fraction of the 500 pairs where the LLM-baseline returns a verifiable `P_llm`.
- **Cost ratio** (averaged across all 500 pairs, mixing hits and misses).
- **Size ratio** on hits vs. misses, reported separately.
- **Minimality.** For `fastpath_only` and `fastpath_then_blip`: on a 50-pair audit of hits, run Sequential_Greedy on the returned `P_llm` and report the fraction where at least one sentence can be removed without breaking equivalence. This quantifies how much minimality the fast-path actually gives up in practice.

Expected pattern, given the paper's Table 1 (LLM-citation accuracy ≈ 65% on gpt-4o-mini averaged across workloads):

- Hit rate ≈ 0.55–0.75 on the `paper` workload (lookup-heavy questions favor the fast-path).
- `fastpath_refine` cost ratio strictly below `Embedding_adaptive_EXP` on hits, comparable on misses, and below the unconditioned baseline overall.
- `fastpath_only` size ratio higher than BLIP's (no minimality), but possibly still within the "useful for human review" range.

### Experiment 8 — (Optional, milestone 2) Top-k provenance

Vary `k ∈ {1, 2, 3, 5}`. Report cost ratio and avg size ratio per `k`. Expectation (paper Fig. 11): cost ratio ~1.3× → ~1.8× as k goes 1 → 5.

## 3.9 Result schema (per pair × strategy)

Every run of a strategy on a pair writes one JSONL row to `runs/<ts>/<strategy>.jsonl`. Each row contains everything needed to recompute any aggregate without re-running the experiment.

```jsonc
{
  // --- identification ---
  "pair_id":          "0001",
  "doc_id":           "https://doi.org/10.1145/3335082.3335100",
  "question_hash":    "a3f1…",         // SHA-256 of the question text
  "strategy":         "Embedding_adaptive_EXP",
  "fastpath_mode":    "refine",         // "off" | "only" | "refine" | "then_blip"
  "run_id":           "20260520-1545-7b3c",
  "git_sha":          "abc1234",
  "timestamp_utc":    "2026-05-20T15:45:01Z",

  // --- inputs ---
  "text_tokens":      5832,             // |T|
  "text_sentences":   412,              // n
  "baseline_answer":  "sustainability", // A = L(T, Q)
  "baseline_answer_tokens": 1,
  "ground_truth":     "sustainability", // from papers.json, optional

  // --- output: the provenance ---
  "provenance_idxs":      [37, 124, 198],   // sentence indices in T
  "provenance_sentences": 3,
  "provenance_tokens":    87,
  "size_ratio":           0.0149,            // provenance_tokens / text_tokens

  // --- accuracy (THE headline metric) ---
  //   accuracy_provenance is the one BLIP guarantees == 1.0 (Theorem 2).
  //   accuracy_answer_vs_gt and accuracy_baseline_vs_gt are recorded so we can
  //   tell whether failures are "LLM was wrong on the full text to begin with"
  //   vs. "BLIP returned a bad P".
  "accuracy": {
    "accuracy_provenance":     true,         // I(A', A)            — does P reproduce A
    "accuracy_answer_vs_gt":   true,         // I(A', ground_truth) — is A' correct
    "accuracy_baseline_vs_gt": true          // I(A,  ground_truth) — was A correct to begin with
  },
  "verified":         true,                  // alias for accuracy.accuracy_provenance
  "final_answer":     "sustainability",      // A' = L(P, Q)
  "final_answer_tokens": 1,

  // --- fast-path ---
  "fastpath_hit":     false,                 // true if the LLM-baseline alone produced P

  // --- llm call counts ---
  "llm_calls": {
    "fastpath_elicit": 1,
    "fastpath_verify": 1,
    "prune":           4,
    "refine":          6,
    "verify_final":    1,                    // the pipeline.run assertion
    "judge":           7,                    // I(A, A') invocations
    "total":          20
  },

  // --- token totals ---
  "tokens": {
    "input_total":        18420,            // sum of prompt_tokens across all calls
    "input_cached_total":  9210,            // sum of cached_tokens (subset of input)
    "output_total":         142,            // sum of completion_tokens
    "by_phase": {
      "fastpath_elicit": { "in":  6800, "in_cached":    0, "out":  35 },
      "fastpath_verify": { "in":  6850, "in_cached": 6800, "out":   2 },
      "prune":           { "in":  2400, "in_cached": 1900, "out":  20 },
      "refine":          { "in":  1810, "in_cached": 1400, "out":  72 },
      "verify_final":    { "in":   200, "in_cached":  110, "out":   2 },
      "judge":           { "in":   360, "in_cached":    0, "out":  11 }
    }
  },

  // --- cost ($USD), via Eq. 1/2 with f_L ---
  "cost_usd": {
    "baseline":  0.00109,                   // cost(answer(Q, T)) — denominator
    "total":     0.00118,                   // numerator
    "by_phase": {
      "fastpath_elicit": 0.00041,
      "fastpath_verify": 0.00026,
      "prune":           0.00018,
      "refine":          0.00011,
      "verify_final":    0.00002,
      "judge":           0.00020
    }
  },
  "cost_ratio": 1.083,                      // cost_usd.total / cost_usd.baseline

  // --- latency (seconds, wall clock) ---
  "latency_s": {
    "total":     8.31,
    "by_phase": {
      "fastpath_elicit": 1.9,
      "fastpath_verify": 1.8,
      "prune":           1.7,
      "refine":          2.3,
      "verify_final":    0.4,
      "judge":           0.2                // judge calls are short
    }
  },

  // --- strategy config snapshot (for reproducibility) ---
  "config": {
    "model_driver":     "gpt-54-mini",
    "model_judge":      "gpt-54-mini",
    "model_embed":      "text-embedding-3-small",
    "block_count_m":    20,
    "refine_threshold_t": 10,
    "f_L_driver":       2,
    "judge_prompt":     "llm_equal_human_example",
    "seed":             42,
    "system_fingerprint_worst": "fp_4eaa3b1"  // worst (last-seen) across all calls in this row
  }
}
```

Notes on what's in here and why:

- **Yes to all four** of {latency, cost, input tokens, output tokens}. Input is decomposed further into cached vs. uncached so Eq. 2 is recomputable from the row alone.
- **By-phase breakdown.** A single number per metric is useless for debugging. When cost ratio is high we need to know whether Refine blew up or the fast-path missed; when latency is high we need to know which phase. The `by_phase` blocks pay for themselves the first time a strategy regresses.
- **`llm_calls` counts.** Distinct from token totals — sometimes the count is what tells you the algorithm misbehaved (e.g., Refine took 30 calls instead of 6).
- **`fastpath_hit`.** Required for Experiment 7. Also lets us compute the conditional cost ratios on hits vs. misses.
- **`final_answer`.** Storing the actual `A'` (not just `verified: true`) makes the spurious-provenance audit (Experiment 6) possible without re-running anything.
- **`system_fingerprint_worst`.** If Azure swaps the underlying model mid-run, we want a way to detect it post-hoc.

Aggregates over the sample go into a separate `summary.jsonl`. The headline rows, in order:

1. **`accuracy.accuracy_provenance` pass rate** — the BLIP guarantee. Must be `1.0`. If anything less, halt and fix before reporting any other number.
2. **`accuracy.accuracy_answer_vs_gt` pass rate** — measures whether BLIP's `A'` is actually correct vs. the human-labeled ground truth. Bounded above by `accuracy.accuracy_baseline_vs_gt` (we can never exceed the LLM's own competence on the full text).
3. **`accuracy.accuracy_baseline_vs_gt` pass rate** — measures whether `gpt-54-mini` is competent on this corpus at all. Reported once per workload, shared across strategies.
4. **`cost_ratio`, `size_ratio`, `latency_s.total`** — mean, p25, p50, p75, 95% bootstrap CI.
5. **`fastpath_hit` rate** — fraction of pairs where the fast-path returned a verified `P`.

## 4. Statistical reporting

- All averages reported with 95% bootstrap CI (1,000 resamples) over the 500 pairs.
- Per-strategy results saved as JSONL under `runs/<timestamp>/<strategy>.jsonl`, one row per pair.
- Aggregation script reads JSONL → summary CSV + Markdown table identical in shape to paper Table 6.

## 5. Judge calibration

Before reporting any results, validate the judge:

1. Sample 100 pairs at random.
2. For each, generate two candidate answers `A_1 = L(T, Q)` and `A_2 = L(T, Q)` with `temperature=0`. With determinism these should match; record disagreement rate.
3. Manually label 50 (`A`, `A'`) pairs where `A ≠ A'` lexically. Compute judge agreement vs. human labels. Target ≥ 90% (paper claims >94% on five LLMs).
4. If agreement < 90%: switch judge model from `gpt-54-mini` to `gpt-54` and re-test.

Calibration must be re-run whenever the judge prompt changes.

## 6. Spurious-provenance audit

For a random 50-pair audit:

- Confirm the returned provenance, read aloud, plausibly supports `A`. This is the "correctness" check (paper §2.5) — distinct from verifiability.
- Record cases where verifiability holds but the provenance does *not* support the answer (spurious). Report the fraction; the paper expects this to be small (single-digit %).

This audit is qualitative and not gating, but it surfaces judge weaknesses.

## 7. Cost-budget guardrails

The default is to *stay* at small samples unless we have a specific reason to scale up. Concretely:

1. **Pilot (20 pairs).** Run all strategies. From the observed per-pair cost, extrapolate the cost of step 2 and step 3 and write the numbers to `runs/<ts>/budget.json`.
2. **Gate before step 2.** Don't run the 50-pair sample unless the pilot's verifiability rate is 1.0 and the cost-ratio numbers look in the right ballpark (≤ 2×). Otherwise: debug, repeat the pilot.
3. **Gate before step 3.** Only run the full 500-pair sample if (a) step 2 numbers are stable across reruns and (b) projected full-sample spend is < $30. Otherwise: stop at 50, or downsample to 200.

There is no obligation to ever run 500 pairs. The reproduction is "done" the moment we have a small-sample number we trust — typically 50 pairs is plenty to make a qualitative claim. The 500-pair sample only matters if we want headline numbers directly comparable to paper Table 6.

## 8. Reproducibility ledger

Every experiment row records:

- Strategy spec (`prune_fn`, `refine_fn`, `ranker`)
- `seed`, `block_count_m`, `judge_prompt`, `judge_model`
- Driver `model` + Azure `system_fingerprint` per call (worst-of)
- BLIP code git SHA
- Timestamp

This is enough to re-run any single number from the result tables.

## 9. Reporting format

For each experiment, produce a table shaped like paper Table 6 (rows = strategies, cols = `size_ratio`, `cost_ratio`, `latency`). Highlight the top-3 by `cost_ratio` and by `latency`. Save to `docs/results/exp_<N>.md` (the `results/` subfolder is created on first run).

## 10. What we are *not* reporting

- Per-question-type breakdown (paper Table 5 categories). Possible follow-on once a coarse question-type classifier is added.
- Skewed-workload comparison (paper Fig. 8b). Requires synthetic skewing; out of scope.
- Cross-LLM comparison (paper Table 10 gemini-2-flash column). Optional; requires a second deployment.
- Human user study (paper §5.3 Table 12). Out of scope.
