# BLIP Reproduction — Overview

> Bolt-on, Verifiable Provenance for LLM-Powered Data Processing
> Lin, Zeighami, Parameswaran (UC Berkeley). Source: `BLIP_revision_TR.pdf` in `Projects/BLIP/`.

## 1. What problem BLIP solves

When an LLM `L` answers a question `Q` over some text `T`, the user sees only the answer `A = L(T, Q)`. There is no indication of *which part of `T`* was actually needed to produce `A`. Retrieval scores and "ask the LLM for citations" both fail in practice — the paper shows accuracy below 0.8 even with top-10% retrieval, and that LLM-produced citations only reproduce the answer in 53–74% of cases (Table 1).

BLIP introduces **verifiable provenance**: a subsequence `P ⊑ T` such that re-running `L(P, Q)` yields an answer equivalent to `A`. The goal is to find a **minimal** such `P` (no sentence in `P` can be removed without breaking equivalence) at low cost, treating `L` as a black box.

## 2. What we are reproducing

Scope: full reproduction of BLIP's core framework.

In scope:

- **Fast-path front door.** Before invoking BLIP, issue one LLM call asking the model to cite its own provenance (the paper's `LLM-Provenance-Prompt` baseline from §1 / Table 1). Verify it with one more call; if it verifies, return it and skip BLIP entirely. On a miss, fall through to the full pipeline. See `04_algorithms.md` §0.
- Phase 1 (Prune): four ranker/scan strategies + adaptive strategy
  - `Embedding_bottom_up`, `Embedding_top_down`
  - `LLM_bottom_up`, `LLM_top_down`
  - `Embedding_adaptive`, `LLM_adaptive`
- Phase 2 (Refine): `Sequential_Greedy` and `Exponential_Greedy`
- Top-k provenance via tree-based search (Algorithm 3)
- Cost model with KV-cache (Eqs. 1–2) and the cost ratio metric
- LLM-as-a-judge for answer equivalence `I(A, A')` (Definition 1)
- The adaptive crossover-point heuristic `CP = (L_m + U_m)/2`

Out of scope for the first pass (left as stretch goals):

- All other paper datasets (NL_DEV, HotpotQA, CUAD, PubMedQA, Movie, Restaurant). Only the local `paper` workload is targeted.
- TableQA recovery metrics `R` / `eR` (not applicable — no ground-truth provenance tuples in the `paper` workload).
- Human user study (Section 5.3 / Table 12).
- Non-OpenAI providers beyond an optional `gemini-2-flash` comparison.

## 3. Workload we will use

**Single workload for this reproduction:** the local `workload/paper/` corpus.

- `query/papers.json` — 584 scientific-paper records, each with `title`, extracted `text`, and a list of `questions` (`{question, answer}` pairs). Total: 7,961 Q–T pairs, averaging ~13.6 questions per paper (min 10, max 17).
- `data/raw_data/` — 102 source PDFs (a subset of the 584; the extracted `text` field in the JSON is what BLIP will actually consume).

This dataset plays the same role Qasper plays in the paper: long scientific text with multiple lookup / aggregation / reasoning questions per document. Other datasets from the paper (NL_DEV, HotpotQA, CUAD, PubMedQA, Movie, Restaurant) are explicitly **out of scope** for this iteration. They can be added later by dropping new loaders into the same workload-adapter interface (see `02_data_workloads.md`).

## 4. LLM stack assumed

We use **Azure OpenAI** as the API provider. The deployment names and key locations are configured in `local/azure.json`:

```json
{
  "key_file":           ".../api_keys/azure_cloudbank/gpt-54_1.txt",          // larger model deployment
  "key_file_cheap":     ".../api_keys/azure_cloudbank/gpt-54-mini.txt",       // cheap model deployment — the driver
  "embedding_key_file": ".../api_keys/azure_cloudbank/embedding3small.txt"    // text-embedding-3-small
}
```

Roles:

- **Driver LLM `L`** (used for both answering `A = L(T, Q)` and verification `L(P, Q)`): the `gpt-54-mini` Azure deployment. This is the analog of `gpt-4o-mini` in the paper and is the cost-sensitive workhorse of the pipeline.
- **LLM ranker `L'`** (only used by `LLM_bottom_up` / `LLM_top_down` / `LLM_adaptive`): also `gpt-54-mini` initially, since it does not need to equal `L`. If ranking quality is poor we can upgrade `L'` to the `gpt-54` deployment.
- **Equivalence judge:** `gpt-54-mini` by default; can be upgraded to `gpt-54` if judge agreement looks weak in spot checks.
- **Embedding model:** `text-embedding-3-small` via the Azure embedding deployment. Replaces `all-mpnet-base-v2` from the paper — same role (offline sentence/block ranking), different vendor.
- **KV-cache:** rely on Azure OpenAI's automatic prompt-prefix caching. The cache speedup factor `f_L` for `gpt-54-mini` (Azure's gpt-4o-mini family) is `f_L = 2` per the paper's Eq. 2 numbers. We treat this as a config knob (`f_L_cheap = 2`) — to be re-measured empirically once we observe real input-token billing.

This is the only stack the reproduction plan assumes. No Gemini, no local open models, no direct OpenAI API — strictly Azure.

## 5. Success criteria

A reproduction is considered successful when, on our `paper` workload with the `gpt-54-mini` Azure deployment as driver:

1. **Verifiability = 100%.** Every provenance returned by BLIP satisfies `I(L(P, Q), A) = True`. This is BLIP's hard guarantee (Theorem 2) — any miss is a bug, not a metric.
2. **Size ratio < 15%** with `Embedding_adaptive_EXP` (paper reports 3.4% on Qasper; we expect somewhat higher on our `paper` corpus because the questions are more lookup-heavy than Qasper's; <15% is a generous bound).
3. **Cost ratio ≈ 1×** of full-text answering (paper: ~1.1× on Qasper, Table 6).
4. **Pruning-only cost ratio < 0.5×** for the prune-only configurations (paper: 0.36× on Qasper for `Embedding_adaptive`).
5. The adaptive strategy matches or beats the better of bottom-up vs. top-down on at least 80% of (question, document) pairs.

## 6. Document layout

This `docs/` folder contains the reproduction plan:

- `00_overview.md` — this file
- `01_problem_definitions.md` — formal definitions, monotonicity, cost model, KV-cache
- `02_data_workloads.md` — local `paper` workload + external benchmarks
- `03_architecture.md` — module layout, interfaces, data flow
- `04_algorithms.md` — Algorithms 1–3 spec, prompts, adaptive crossover
- `05_evaluation_plan.md` — metrics, experiments, baselines, judges
- `06_roadmap.md` — phased milestones, risks, deliverables
- `07_testing.md` — unit / integration / live / reproduction tests

## 7. Why this is worth doing carefully

BLIP's correctness proof (Theorems 1–3) only kicks in if (a) the equivalence judge `I` is correct and (b) the prune+refine pipeline actually verifies every candidate provenance before returning it. A naive implementation that, e.g., skips the post-prune refinement, or uses a sloppy judge prompt, will silently violate minimality without throwing any error. The plan below treats the judge and the verification calls as first-class components and tests them in isolation before composing the pipeline.
