# Problem Definitions

All numbering follows the paper. Re-read this whenever an algorithm decision feels ambiguous — most ambiguities trace back to one of these definitions.

## 1. Setting

- `L` — a black-box LLM. In this reproduction `L` is the Azure deployment `gpt-54-mini` (the deployment maps to the gpt-4o-mini model family).
- `T = ⟨s_1, s_2, …, s_n⟩` — the source text, a sequence of sentences. Order matters for KV-cache reuse and for output stability.
- `Q` — a natural-language question.
- `A = L(T, Q)` — the LLM's answer over the full text.
- `P ⊑ T` — a subsequence of `T`, i.e., `P = ⟨s_{i_1}, …, s_{i_m}⟩` with `1 ≤ i_1 < … < i_m ≤ n`. Subsequences preserve order; they are *not* arbitrary subsets.
- `I(A, A')` — indicator returning `True` iff `A` and `A'` are "the same" answer. Implemented either as exact string match (closed-domain answers like yes/no, year, zip code) or as an LLM-as-a-judge (open-domain). See §6.

## 2. Verifiable provenance (Definition 1)

`P ⊑ T` is a **verifiable provenance** for `⟨T, Q, L⟩` and answer `A` if

```
I(L(P, Q), A) = True
```

In words: running the LLM on just `P` produces an equivalent answer to running it on the full `T`. "Equivalent" — not "identical" — because LLM outputs can vary in phrasing even at temperature 0.

Throughout the paper "provenance" without qualifier means "verifiable provenance".

## 3. Minimal vs. strictly minimal provenance (Definitions 2 & 3)

- **Minimal provenance.** `P` is minimal iff `∀ s ∈ P: I(L(P \ s, Q), A) = False`. Removing *any single* sentence breaks equivalence.
- **Strictly-minimal provenance.** `P` is strictly minimal iff no subsequence `P' ⊏ P` is itself a provenance. This is a stronger property than minimal: no subset, not just no single-sentence removal, can be a provenance.

Strictly-minimal ⟹ minimal, never the other way. In practice BLIP returns minimal provenance, and the paper shows (Theorem 1) that for *weakly-monotonic* tasks, minimal ⟺ strictly-minimal.

## 4. Monotonicity (Definitions 4 & 5)

A task `⟨Q, T, L⟩` characterizes how provenance behaves under supersets. The intuition: "if `P` is a provenance, does adding more text keep it a provenance?"

- **Strongly monotonic.** For any provenance `P ⊑ T`, *every* superset `P' ⊒ P` with `P' ⊑ T` is also a provenance. Empirically present in ~90% of cases across the paper's workloads (Table 3).
- **Weakly monotonic.** For any provenance `P ⊑ T` of size `i`, there exists a *chain* of provenances `P_{i+1}, P_{i+2}, …, P_{n-1}` with `|P_{k}| = k` and `P ⊑ P_{i+1} ⊑ P_{i+2} ⊑ … ⊑ T`. Empirically present in ~95% of cases.

Implication for our reproduction: we do **not** need to special-case non-monotonic tasks. The pruning algorithms still return *a* minimal provenance even when monotonicity fails — they may just miss alternative minimal provenances.

## 5. The two problems BLIP solves

- **Problem 1 (Provenance Inference).** Given `⟨T, Q, L, A⟩`, return *one* minimal provenance.
- **Problem 2 (k-Provenance Inference).** Given `⟨T, Q, L, A⟩` and integer `k`, return up to `k` distinct minimal provenances, no two equal.

Problem 1 is solved by Phase 1 (Prune) + Phase 2 (Refine). Problem 2 wraps Problem 1 in a tree-search (Algorithm 3). For the first reproduction milestone we only need Problem 1; Problem 2 is added once the core pipeline passes tests.

## 6. Equivalence judge `I(A, A')`

The judge is the single most safety-critical component. A wrong judge silently breaks both verifiability and minimality.

- **Closed-domain (yes/no, dates, zip codes, enum):** use exact string match after light normalization (lowercase, strip punctuation, collapse whitespace).
- **Open-domain:** LLM-as-a-judge. The paper reports >94% agreement with humans (Table 2). They evaluate three prompt variants and find they all work; we'll standardize on `llm_equal_human_example` (two-shot) for the reproduction since it had the best aggregate numbers across the five LLMs they tested. The judge model is the `gpt-54-mini` Azure deployment by default.
- **Probability-tie relaxation (paper §2.6).** Strictly, `I` should compare the set of max-probability outputs `𝒜` and `𝒜'`, with ε = 0.01 tolerance for floating-point ties. In practice this matters in <1% of cases. We will implement greedy (temperature=0) sampling once per call and skip the multi-sample relaxation in the first milestone, documenting it as a known simplification.

The judge LLM does not need to be the same as `L`. We use `gpt-54-mini` for both to keep cost down; if results drift, switch the judge to the larger `gpt-54` deployment.

## 7. Cost model (Eqs. 1–2)

Let `P_T = ⟨Q, T⟩` denote the concatenated prompt. The uncached cost of one LLM call:

```
c_{PT} = |P_T| · c_in  +  |L(P_T)| · c_out               (1)
```

with `|·|` measured in tokens, `c_in` and `c_out` from the provider's price sheet.

When two prompts `P_{T_i}` and `P_{T_j}` share a prefix `Pre_{i,j}`, KV-cache reduces the input cost of the *second* call:

```
c_{PT_j | PT_i} = (|Pre_{i,j}| / f_L) · c_in
                  + |P_{T_j} \ Pre_{i,j}| · c_in
                  + |L(P_{T_j})| · c_out                  (2)
```

`f_L` is the per-model cache cost-reduction factor:

| Deployment (Azure)           | `c_in` ($/1M tok)             | `c_out` ($/1M tok)            | `f_L` (assumed) |
| ---------------------------- | ----------------------------- | ----------------------------- | --------------- |
| `gpt-54-mini` (driver)       | per Azure price sheet (small) | per Azure price sheet (small) | 2               |
| `gpt-54` (escalation)        | per Azure price sheet (large) | per Azure price sheet (large) | 2               |
| `text-embedding-3-small`     | per Azure price sheet         | n/a (embedding)               | n/a             |

`f_L = 2` is the value the BLIP paper uses for the gpt-4o-mini family and is our starting assumption for `gpt-54-mini`. We will re-measure `f_L` empirically after the first end-to-end run (observe `prompt_tokens_cached` in Azure usage objects) and update the cost-ratio computation accordingly.

The cost model drives algorithm design — every strategy is built so that consecutive LLM calls share the longest possible prompt prefix.

## 8. Cost ratio and size ratio (metrics)

For a given strategy `S` applied to `⟨T, Q, L, A⟩` returning provenance `P`:

- **Size ratio.** `|P| / |T|` (in tokens). Lower = smaller provenance.
- **Cost ratio.** `cost(S) / cost(answering Q on full T)`. Numerator includes every LLM call made by `S` (ranker + verifier + refine). Denominator is one call: `c_{Q,T}`. The paper convention is that the *embedding* cost is excluded from cost ratio because embeddings are computed offline and re-used across questions for the same document.
- **Latency.** Wall clock for the strategy, parallelizing where the paper does (none of the algorithms parallelize across questions for the *same* document, but embedding-rank precomputation is offline).

For prune-only configurations, cost ratio counts only the prune-phase calls. For two-phase configurations, it counts prune + refine.

## 9. Correctness of provenance (§2.5)

A verifiable provenance is **correct** if it enables a human to arrive at the same answer as the LLM did. This is a *semantic* property the paper distinguishes from verifiability:

- **Unstable provenance.** Question is ambiguous → no well-defined true provenance → verifiable provenance may still be misleading.
- **Spurious provenance.** LLM judge `I` incorrectly says `I(L(P, Q), A) = True` when the answer on `P` is actually different.

For our reproduction we accept the paper's stance: BLIP guarantees verifiability, not correctness. Correctness is the user's responsibility. We do report any judge-disagreement cases caught by sanity checks (see `05_evaluation_plan.md` §6).

## 10. Scope contract

This reproduction targets exactly the definitions above on the local `paper` workload. No SPJRU / `RA+` extensions (paper §3.3.5), no SQL rewriting, no tracking provenance through aggregations beyond what naturally falls out of `I(A, A')`.
