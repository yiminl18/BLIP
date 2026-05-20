# Algorithms

Pseudocode-level spec for Algorithms 1–3 from the paper, plus the adaptive strategy, the crossover-point heuristic, and all prompt templates. Read this alongside the algorithm boxes in the paper (pages 7–8 and the top-k box on page 11).

Notation throughout: `T = ⟨s_1, …, s_n⟩`, blocks `B = ⟨b_1, …, b_m⟩` with `m = 20`, question `Q`, target answer `A = L(T, Q)`, equivalence `I(·, ·)`. All `L(P, Q)` invocations are made with sentences in `P` placed in their original `T` order (this is enforced by `pipeline.py`).

## 0. LLM-baseline fast-path (front door)

Before running the BLIP prune+refine pipeline, we cheaply probe whether the LLM can identify its own provenance on this question. Concretely: issue one `LLM-Provenance-Prompt` call (the same baseline the paper compares against in §1 and Table 1) and verify the returned sentence list. If it verifies, return it and stop. Otherwise, fall through to Phase 1 (Prune).

```
function blip_with_fastpath(T, Q, A, L):
    # one cheap call: ask L to cite its own provenance
    sentence_ids ← L(LLM_provenance_prompt(Q, A, T))      # 1 LLM call
    P_llm       ← parse_sentence_ids(sentence_ids) ∩ valid(T)
    if P_llm is empty:
        return blip(T, Q, A, L)                            # nothing usable

    # one cheap call: verify
    A'          ← L(reorder(P_llm), Q)                     # 1 LLM call
    if I(A, A'):
        return P_llm                                       # fast-path hit
    return blip(T, Q, A, L)                                # fall through
```

Cost accounting. The fast path adds **2 LLM calls** on every question: one to elicit `P_llm`, one to verify it. On a *hit*, BLIP saves all of Phase 1 and Phase 2 (typically 6–30 calls), so even at the paper's reported ~65% hit rate this is a net win. On a *miss*, we pay 2 extra calls relative to plain BLIP.

Why this works as a prefilter and not as a primary solution. Paper Table 1 shows that LLM-citation alone gives ~65% / 74% / 53% accuracy on gpt-4o-mini / gpt-4o / gemini-2-flash — i.e., 26–47% of cases produce a non-verifiable citation. BLIP-with-fastpath inherits the *speed* of LLM-citation on the easy questions and the *guarantees* of BLIP on the hard ones.

What the fast-path gives up. The returned `P_llm` is verifiable but **not guaranteed minimal**. BLIP's minimality theorem (Theorem 2) only applies to provenances that go through Refine. If minimality matters for the downstream use (e.g., short snippets for human review), use one of the variants in §0.1. If only verifiability matters (e.g., trust signal that "yes, the answer is grounded in this text"), the bare fast-path is enough.

### 0.1 Variants on the fast-path

Three configurations, in order of cost vs. guarantee:

| Variant                 | On hit, returns         | Minimal? | Extra cost on hit |
| ----------------------- | ----------------------- | -------- | ----------------- |
| `fastpath_only`         | `P_llm` as-is           | No       | 2 LLM calls       |
| `fastpath_refine`       | `Refine(P_llm)`         | Yes      | 2 + Refine cost   |
| `fastpath_then_blip`    | `P_llm` (no refine)     | No       | 2 LLM calls       |

`fastpath_only` and `fastpath_then_blip` differ only in their *miss* path (the table covers the *hit* path). On a miss they both fall through to full BLIP.

`fastpath_refine` is interesting: skip Phase 1 entirely on a hit, run Phase 2 on `P_llm`. This preserves minimality and still saves the ~3–5 prune-phase LLM calls per question. Cost ratio analysis: `cost(fastpath_refine | hit) ≈ 2/n_full + cost(Refine on |P_llm|) / cost_full`. With `|P_llm|` typically small, this can match or beat `Embedding_adaptive_EXP`.

**Default for this reproduction:** `fastpath_refine`. It buys back the minimality guarantee at marginal extra cost, and lets us still report cost-ratio vs. paper Table 6.

### 0.2 Sentence-ID parser

The `LLM-Provenance-Prompt` (§5.3) instructs the LLM to return a list of sentence IDs. In practice the output varies:

- `"3, 7, 12"` — clean comma-separated list (target format).
- `"[3, 7, 12]"` — bracketed list.
- `"Sentence IDs: 3, 7, 12"` — preamble + list (model ignored "do not add explanations").
- `"3\n7\n12"` — newline-separated.
- `"None"` / `""` — model claims nothing in `T` supports `A`.

The parser:

1. Strip everything up to the first digit.
2. Extract all integer tokens via `re.findall(r"\d+")`.
3. Filter to `1 ≤ id ≤ n`; deduplicate; preserve order.
4. If the result is empty, treat as "no provenance" and fall through to full BLIP (do not skip).

We pre-number sentences in the prompt as `[1] s_1\n[2] s_2\n…` so the model has an unambiguous numbering to cite.

### 0.3 KV-cache angle

Both fast-path calls share their `[Question]` and `[Context]` portions with the eventual full-BLIP calls. We use the same SYSTEM message and the same `Text:\n"""…"""\n\nQuestion: …` envelope for the elicitation call, the fast-path verification call, and every downstream `L(P, Q)` call. So a *miss* still primes the KV cache for the prune phase — the wasted 2 calls are not as wasted as they look.

### 0.4 Where it sits in the pipeline

The fast-path is implemented as a wrapper in `pipeline.py`, not inside any algorithm module. Existing prune/refine modules are untouched. Setting `Config.fastpath = "off" | "only" | "refine" | "then_blip"` selects the variant.

## 1. Algorithm 1 — `Prune[ranker, scan]`

### 1.1 Bottom-up scan

```
function prune_bottom_up(T, Q, A, L, ranker):
    B           ← split T into m=20 equal-sized blocks    # contiguous, ordered
    order       ← ranker.rank(B, Q, A)                    # indices, most-relevant first
    P           ← ∅                                       # accumulated indices
    for j in order:                                       # ascending priority
        P       ← P ∪ {j}
        P_text  ← reorder(P)                              # reorder back to T order
        A'      ← L(P_text, Q)
        if I(A, A'): break
    return rerank_to_T_order(P)                           # final sort by sentence idx
```

Notes:
- `rerank_to_T_order` is critical. The paper calls it `rerank_index` (Line 7 of Algorithm 1). Reason: if blocks are fed to `L` in ranker order rather than `T` order, the meaning can change (e.g., "the previous experiment" loses its referent), which can break the answer and inflate the prune size.
- Worst case: all 20 LLM verifier calls if the provenance is in the last block. Average is much smaller; the paper's `Embedding_bottom_up` averages ~10.5% size ratio on Qasper.

### 1.2 Top-down scan

```
function prune_top_down(T, Q, A, L, ranker):
    B           ← split T into m=20 blocks
    order       ← ranker.rank(B, Q, A)
    P_full      ← rerank_to_T_order(B)                    # the entire T
    assert I(A, L(P_full, Q))                             # sanity: full T must verify
    l, r        ← 1, m
    while l ≤ r:
        mid     ← ⌈(l + r) / 2⌉
        candidate ← top-mid blocks of `order`, sorted to T order
        A'      ← L(candidate, Q)
        if I(A, A'):
            best ← candidate
            r   ← mid − 1                                  # try smaller
        else:
            l   ← mid + 1                                  # need more
    return rerank_to_T_order(best)
```

Notes:
- Binary search over `i` ∈ [1, m]. At most `⌈log₂ m⌉ = 5` LLM verifier calls when `m=20`. Wins when the provenance is large.
- The full-`T` sanity check at the top is editorial — it is *not* in the paper's Algorithm 1 box, but without it top-down can wrongly conclude the provenance is empty if the LLM is non-deterministic on the full input.

### 1.3 Choosing between `embedding` and `llm` rankers

Same Prune skeleton, only `ranker` differs. EmbeddingRanker is free at runtime (precomputed) but less precise on hard rankings; LLMRanker costs one LLM call per pair but is sharper. See `03_architecture.md` §6 for the abstraction.

## 2. Algorithm 2 — Refine

### 2.1 `Sequential_Greedy`

```
function sequential_greedy(P, T, Q, A, L):
    changed ← True
    while changed:
        changed ← False
        for s in P sorted by sentence index DESCENDING:    # high-index first → max KV reuse
            P'   ← P \ {s}
            A'   ← L(reorder(P'), Q)
            if I(A, A'):
                P  ← P'
                changed ← True
    return P
```

KV-cache reasoning (paper Example 1, §3.2): if we remove sentences in *descending* index order, the prompt prefix `⟨s_1, …, s_{k-1}⟩` is reused across consecutive calls until we hit `s_k` itself. With `f_L = 2` on `gpt-54-mini`, the cached tokens cost half. SEQ does linear scans, possibly multiple passes — but those later passes also share prefixes because we always re-process descending.

### 2.2 `Exponential_Greedy`

```
function exponential_greedy_inner(P, T, Q, A, L):
    j     ← |P| - 1                  # 0-based last index
    l     ← 0                        # exponent
    while j ≥ 0:
        i ← max(0, j - 2^l + 1)
        chunk ← P[i:j+1]             # candidate to remove (contiguous in P-order)
        P'    ← P \ chunk
        A'    ← L(reorder(P'), Q)
        if I(A, A'):
            P ← P'
            j ← i - l                # paper Algorithm 2 line 14: j ← i − 1 − l + 1
            l ← l + 1                # grow next window
        else:
            l ← 0                    # reset window; j stays put — next iter retries with single-sentence window
    return P

function exponential_greedy(P, T, Q, A, L):
    P_prev ← None
    while P_prev != P:
        P_prev ← P
        P ← exponential_greedy_inner(P, T, Q, A, L)
    return P
```

Notes:
- The outer loop mirrors paper Algorithm 2 lines 22–26: after one pass of `Exponential_Greedy`, re-apply it until the provenance stops changing. The paper does **not** finish EXP with a SEQ pass — both branches (SEQ and EXP) self-repeat.
- "Useless probe" (paper §3.3.3): on a failed removal we keep `j` the same and only reset `l ← 0`. The next iteration retries the removal with a window of size 1 (since `2^0 = 1`). Decrementing `j` on failure would over-step and miss sentences.
- Asymptotics (paper §3.3.3): for sequences with `g = l − h` non-provenance sentences, `C_EXP = Θ(g log² g + h log² g + h²)` vs. `C_SEQ = Θ(gl + h²)`. Practically, EXP wins when `> 10` sentences enter Phase 2.

### 2.3 Two-phase glue

```
function single_minimal_provenance(pair, ranker, scan, refine):
    pruned   ← prune(pair, ranker, scan)
    minimal  ← refine(pruned, pair)
    assert verify(minimal, pair)
    return minimal
```

`refine = sequential_greedy` if `|pruned| < t` else `exponential_greedy`. Default `t = 10` per paper §5.2.

## 3. Adaptive prune

### 3.1 Crossover-point heuristic

From paper §3.3.2 and Theorem 4:

```
L_m = (√(8m - 7) - 1) / 2
U_m = (-1 + √(1 + 8(m log₂ m - m + 1))) / 2
CP  = (L_m + U_m) / 2
```

For `m = 20`:

- `L_20 = (√(8·20 − 7) − 1) / 2 = (√153 − 1) / 2 ≈ 5.68`
- `U_20 = (−1 + √(1 + 8·(20 log₂ 20 − 20 + 1))) / 2 ≈ 11.12`
- `CP = (L_20 + U_20) / 2 ≈ 8.40`

This matches the empirical `CP_1 ≈ 8.2` reported in the paper's Figure 9 (Qasper) and Table 9. We pin `CP = 8` for `m = 20` in code.

`CP` answers: "if the final returned provenance occupies fewer than `CP` of the `m` ranked blocks, prefer bottom-up; otherwise prefer top-down."

### 3.2 Adaptive algorithm

```
function adaptive_prune(T, Q, A, L, ranker):
    B           ← split T into m=20 blocks
    order       ← ranker.rank(B, Q, A)
    # 1. probe bottom-up over the top-CP ranked blocks
    P           ← ∅
    for j in order[:CP]:
        P       ← P ∪ {j}
        A'      ← L(reorder(P), Q)
        if I(A, A'): return rerank_to_T_order(P)
    # 2. bottom-up exhausted CP blocks without success → switch to top-down
    #    on the remainder
    return prune_top_down(T, Q, A, L, ranker)            # binary search over all m
```

Notes:
- The paper's wording is "switch to top-down to complete the search" — meaning the binary-search step considers all `m` blocks, not just the remaining `m − CP`. We do likewise.
- Empirically the adaptive strategy ties or beats the better of bottom-up/top-down on most pairs (paper Figure 8). Our success criterion #5 in `00_overview.md` checks this.

## 4. Algorithm 3 — Top-k minimal provenance

Used for the optional Problem 2 (k > 1).

```
function top_k_provenance(T, Q, A, L, ranker, k):
    H ← max-heap                              # (score, block-range) pairs
    H.push((1.0, T[1..m]))                    # full text, score 1
    P_results ← []
    while H not empty:
        score, T_lr ← H.pop()
        if span(T_lr) > 1:                    # node has > 1 block
            mid ← ⌊(l + r) / 2⌋
            score_left  ← topk_eval(T[l..mid], Q, A, L)
            score_right ← topk_eval(T[mid+1..r], Q, A, L)
            c1 ← (l == r AND score ≥ 0.5)
            c2 ← (l < r AND score_left < 0.5 AND score_right < 0.5)
            if c1 OR c2:
                cur ← refine(prune(T_lr, Q, A, L, ranker), Q, A, L)
                if user_verify(cur) == Positive:
                    return cur                # early exit on user approval
                P_results ← P_results ∪ {cur}
            if NOT c2:
                H.push((score_left,  T[l..mid]))
                H.push((score_right, T[mid+1..r]))
        if |P_results| ≥ k: break
    return P_results
```

`topk_eval` uses the `Top-k-Eval-Prompt`:

> Given the following question, [Q], and two answers, [A_1] and [A_2]. Determine whether the two answers are equivalent in meaning. Return the result as a JSON object using the following format: `value: true if the answers are equivalent, false otherwise. score: a real number between 0 and 1 representing the likelihood that your judgment is correct.`

`A_1 = L(T_lr, Q)`, `A_2 = A`. The `score` is what feeds the heap priority.

Top-k is **out of scope for milestone 1** but the module skeleton exists. Implementation gated behind a feature flag.

## 5. Prompts

All prompts live in `src/blip/llm/prompts.py`. They are versioned strings (`v1`, `v2`, …) so an experiment row records exactly which prompt produced it.

### 5.1 Answer prompt (used to compute `A` and to verify `L(P, Q)`)

```
SYSTEM:
You are a careful research assistant. Answer the question using ONLY the
information in the provided text. If the answer cannot be determined from the
text, reply exactly: "I cannot find the answer in the provided text."
Keep answers as short as possible while being complete.

USER:
Text:
"""
{text}
"""

Question: {question}

Answer:
```

Same prompt for `A = L(T, Q)` and for `L(P, Q)`. Reusing the same template is what enables KV-cache to actually trigger — the system message and `Text:` prefix must be byte-identical across calls.

### 5.2 LLM-Ranker-Prompt (paper §3.1)

```
USER:
Given the following question: {question}, and a list of text blocks, the
corresponding answer is {answer}. Your task is to assign a score (from 1 to 10)
to each block based on how likely it is to contain context relevant to
answering the question. The text blocks are listed below, each starting with
Block i: followed by its content. Return only a comma-separated list of scores
corresponding to each block, in the order they appear. Do not include any
explanations or additional text.

{blocks}
```

Output parser tolerates trailing whitespace, missing trailing comma, and extra newlines. On parse failure: fallback to embedding ranker for that pair (logged).

### 5.3 LLM-Provenance-Prompt (paper §1 baseline only)

```
USER:
Given the following question: {question}, the corresponding answers are
{answer}. Your task is to extract the set of sentences from the provided
context that contribute to generating these answers. Identify the most
relevant sentences that support the given answers. Do not add explanations.
Only return a list of sentence IDs. Do not return any words. The context is
as follows:

{numbered_sentences}
```

Used only for the LLM-citation baseline in evaluation, not for BLIP itself.

### 5.4 Judge prompts (`llm_equal_human_example`, two-shot)

```
USER:
Are the following two sentences semantically equivalent? Please respond with
True if they are, and False if they are not.

Examples:
Example 1: Sentence 1: "The cat is sleeping on the sofa." Sentence 2: "A cat
is lying on the couch asleep." Answer: True
Example 2: Sentence 1: "The company reported a profit of 2 million dollars."
Sentence 2: "The company reported a loss of 2 million dollars." Answer: False

Sentence 1: {a}
Sentence 2: {b}

Answer:
```

Parse: strip whitespace, lowercase first token, accept `true`/`false`. On parse failure: one retry; on second failure return `False`.

### 5.5 Top-k-Eval-Prompt

(See §4 above — full text reproduced from paper §4 Algorithm 3.)

## 6. KV-cache discipline

The prompts above are designed so that for a fixed `(Q, A)`:

- The **system message** is invariant.
- The **header** (`Text:\n"""`) is invariant.
- The **suffix** (`"""\n\nQuestion: {Q}\n\nAnswer:`) is invariant.

Only the text body changes. Within a single Refine pass, when we delete sentences in *descending* index order, the prompt prefix from the system message through the last surviving low-indexed sentence is byte-identical to the previous call — which is exactly the precondition for Azure prompt caching to hit.

For SEQ and EXP this is automatic from the iteration order. For Prune-bottom-up the prefix shared across calls is the system message + header (the body grows monotonically), so the cached portion is small but non-zero.

## 7. Determinism and stability

- `temperature=0`, `top_p=1`, fixed `seed`. We log `system_fingerprint` per call. If `system_fingerprint` changes mid-experiment, we abort the run.
- We do not implement the paper's ε-relaxation (§2.6) for tied max-probability outputs. If we see a question with high judge-flip rate across reruns, we either flag it or switch its judge model to the `gpt-54` deployment.

## 8. Failure modes and what to do

| Symptom                                    | Likely cause                                | Fix                                                                |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------------ |
| `verified == False` in final assertion     | Bug in scan order / forgot rerank-to-T      | Inspect `provenance_idxs` ordering; rebuild text via `reorder()`   |
| Refine pass loops forever                  | Judge oscillates between True/False         | Cache judge results within a single Refine run; treat as fixed     |
| Cost ratio > 5×                            | Forgot to share prompt prefix               | Confirm system+header byte-equality across calls                   |
| Size ratio == 1.0 frequently               | Ranker outputs random order                 | Check embedding shape; fall back to LLM ranker; spot-check by hand |
| LLM ranker returns malformed scores        | `Mistral-7B` no longer used; LLM lazy parse | Tolerant parser; on persistent failure switch to embedding ranker  |
