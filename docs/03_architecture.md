# System Architecture

This document defines the module layout, interfaces, and data flow for the BLIP reproduction. The goal is a clean separation between (a) the LLM I/O layer, (b) the ranker layer, (c) the prune/refine algorithms, and (d) the evaluation harness — so that any one of them can be swapped without touching the others.

## 1. High-level data flow

```
                  ┌───────────────────────┐
   workload/paper │ WorkloadAdapter       │  ← papers.json, raw_data/
                  │  iter_pairs()         │
                  │  sentences(doc_id)    │
                  └────────────┬──────────┘
                               │ Pair{doc_id, Q, gt, A_cached}
                               ▼
                  ┌──────────────────────────────┐
                  │ Pipeline                     │
                  │                              │
                  │  ┌────────────────────────┐  │
                  │  │ Fast-path (front door) │  │  ← LLM-Provenance-Prompt + verify
                  │  │  on hit  → return P    │  │     (2 LLM calls; skips BLIP)
                  │  │  on miss → BLIP        │  │
                  │  └────────────┬───────────┘  │
                  │               │ miss          │
                  │  ┌────────────▼───────────┐  │
                  │  │ Phase 1 (Prune)        │  │
                  │  │ Phase 2 (Refine)       │  │
                  │  │ [optional] Top-k       │  │
                  │  └────────────────────────┘  │
                  └────┬─────────┬───────────────┘
                       │         │
            uses ranker│         │uses LLM + judge
                       ▼         ▼
              ┌──────────────┐ ┌──────────────────┐
              │ Ranker       │ │ LLMClient        │
              │ • Embedding  │ │ • answer(prompt) │
              │ • LLM-scored │ │ • verify(P,Q,A)  │ ← judge lives here
              └──────────────┘ └────────┬─────────┘
                                        ▼
                              ┌────────────────────┐
                              │ Azure OpenAI       │
                              │  gpt-54-mini, …    │
                              │  embedding-3-small │
                              └────────────────────┘
```

A single (question, document) pair flows top-to-bottom; the pipeline emits a `ProvenanceResult` containing the returned `P`, every LLM call's token usage, and timing. The fast-path is a wrapper — it never sees the inside of the prune/refine modules.

## 2. Package layout

```
src/blip/
├── __init__.py
├── config.py                # loads local/azure.json, model deployment names
├── workloads/
│   ├── base.py              # WorkloadAdapter protocol + Pair dataclass
│   └── paper.py             # adapter for workload/paper/
├── text/
│   ├── segmenter.py         # pysbd wrapper, sentence list
│   ├── blocks.py            # build m=20 equal-sized blocks from sentences
│   └── tokens.py            # tiktoken wrapper, token counting
├── llm/
│   ├── client.py            # AzureOpenAI wrapper + retry/backoff
│   ├── usage.py             # Usage{prompt_tokens, cached_tokens, completion_tokens}
│   ├── prompts.py           # all prompt templates (answer, ranker, judge, top-k)
│   └── judge.py             # I(A, A'): exact-match or LLM-as-a-judge
├── rank/
│   ├── base.py              # Ranker protocol (rank(items, Q, A) -> ordering)
│   ├── embedding.py         # text-embedding-3-small ranker
│   └── llm.py               # LLM_ranker_prompt scorer
├── algo/
│   ├── prune.py             # Algorithm 1: Prune[ranker, scan]
│   ├── refine.py            # Algorithm 2: Sequential_Greedy, Exponential_Greedy
│   ├── topk.py              # Algorithm 3: top-k via max-heap tree search
│   ├── adaptive.py          # crossover-point heuristic CP
│   └── fastpath.py          # LLM-baseline fast-path (front door)
├── cost/
│   ├── model.py             # Eq. 1 and Eq. 2 cost computation
│   └── accounting.py        # CostLedger: aggregate Usage events
├── pipeline.py              # one-call wrapper: pair → ProvenanceResult
├── cache/
│   └── disk.py              # on-disk memoization for embeddings, answers, tokens
└── runner/
    ├── precompute.py        # offline embeddings + L(T,Q) precomputation
    └── experiment.py        # run a strategy on the 500-pair sample, write JSONL
```

Tests live in `tests/` mirroring this tree. Notebooks (for ad-hoc inspection only) live in `notebooks/` and are gitignored.

## 3. Core data types

```python
@dataclass(frozen=True)
class Sentence:
    idx: int                 # position in T (0-based)
    text: str
    token_count: int

@dataclass(frozen=True)
class Block:
    idx: int                 # 0..m-1
    sentence_idxs: tuple[int, ...]   # indices into T
    text: str
    token_count: int

@dataclass(frozen=True)
class Pair:
    pair_id: str
    doc_id: str
    question: str
    ground_truth: str | None
    llm_answer: str          # A = L(T, Q), cached
    sentences: tuple[Sentence, ...]
    blocks: tuple[Block, ...]

@dataclass
class Usage:
    prompt_tokens: int
    cached_tokens: int       # subset of prompt_tokens served from KV-cache
    completion_tokens: int
    model: str

@dataclass
class ProvenanceResult:
    pair_id: str
    strategy: str            # e.g., "Embedding_adaptive_EXP"
    provenance_idxs: tuple[int, ...]   # sentence indices, sorted ascending
    size_ratio: float
    cost_ratio: float
    latency_s: float
    usages: list[Usage]
    verified: bool           # final I(L(P,Q), A) check
```

`provenance_idxs` is always stored *sorted by sentence index* (i.e., reading order). This is required so that the verifier prompt rebuilds `P` in the same order as it appears in `T` — both for KV-cache reuse and to preserve any answer-stability that depends on text order.

## 4. LLM client

`AzureOpenAIClient` wraps the Azure SDK with three concerns the rest of the system relies on:

1. **Deployment selection.** Methods take a logical model name (`"driver"`, `"escalation"`, `"embed"`) and the client maps to the Azure deployment name from `local/azure.json`. This keeps deployment names out of business logic.
2. **Determinism.** All chat calls use `temperature=0`, `top_p=1`, fixed `seed` where supported. We log `system_fingerprint` per call so we can detect provider-side drift.
3. **Usage reporting.** Every call returns `(content, Usage)`. `cached_tokens` comes from the Azure response's `prompt_tokens_details.cached_tokens` field. The cost-ledger consumes these directly.

Retry policy: exponential backoff on 429 and 5xx, max 5 attempts, jitter. Hard fail on `context_length_exceeded` — that means we tried to feed a prompt that won't fit, and we want a loud error, not a silent truncation.

## 5. Judge `I(A, A')`

`judge.py` exposes one function:

```python
def equivalent(a: str, b: str, *, closed_domain: bool | None = None) -> bool: ...
```

Decision logic:

- If `closed_domain=True` (caller knows the answer space is small/structured): normalize both strings (lowercase, strip punctuation, collapse whitespace) and compare for exact match.
- Otherwise: invoke the LLM judge with the `llm_equal_human_example` two-shot prompt and parse the boolean response. Single retry on parse failure; on second failure default to `False` (be conservative — false negatives lose minimality, false positives lose verifiability, and we prefer the former).

The judge is the only place LLM-as-a-judge runs. Algorithms never directly call the LLM for equivalence — they go through `equivalent(...)`.

Closed-domain heuristic for the `paper` workload: questions whose ground-truth answer is one of `{"Yes", "No"}`, a 4-digit year, or a short comma-separated list of <= 3 tokens are treated as closed-domain. This is a starting heuristic — refine after first eval pass.

## 6. Ranker abstraction

```python
class Ranker(Protocol):
    name: str                # "embedding" | "llm"
    def rank(self, items: Sequence[Block], pair: Pair) -> list[int]:
        """Return item indices in *descending* relevance to (Q, A)."""
```

Two implementations:

- **EmbeddingRanker.** Score each block by `cosine(embed(b), embed(Q + " " + A))`. Embeddings are cached on disk per (doc, block) and per (Q, A). One API call per uncached embedding; offline-precomputable.
- **LLMRanker.** One LLM call that takes all `m=20` blocks at once and returns a comma-separated list of scores 1–10 (paper's `LLM_ranker_prompt`). One call per (pair, prune iteration) — much cheaper than scoring blocks individually.

Both produce a permutation of block indices. The Prune algorithm consumes this permutation; it does not see scores.

## 7. Algorithm modules

Each algorithm module is purely functional — no global state, no I/O, no logging. They take a `Pair`, a `Ranker`, and an `LLMClient`, and return either a smaller block/sentence list (Prune) or a final `ProvenanceResult` (Refine).

- `prune.py: prune(pair, ranker, scan, llm) -> list[int]` — returns the sentence indices kept after Phase 1. `scan ∈ {"bottom_up", "top_down"}`.
- `refine.py: sequential_greedy(p, pair, llm) -> list[int]` and `exponential_greedy(p, pair, llm) -> list[int]`.
- `adaptive.py: adaptive_prune(pair, ranker, llm) -> list[int]` — runs bottom-up on the top-`CP` blocks, top-down on the remainder if needed.
- `topk.py: top_k(pair, ranker, llm, k) -> list[list[int]]` — Algorithm 3.

Composition lives in `pipeline.py`:

```python
def run(pair: Pair, strategy: StrategySpec, ranker: Ranker, llm: LLMClient) -> ProvenanceResult:
    t0 = time.perf_counter()

    # 0. Optional fast-path (LLM-baseline prefilter).
    if strategy.fastpath != "off":
        p_llm = fastpath.elicit(pair, llm)                    # 1 LLM call
        if p_llm and verify(p_llm, pair, llm):                # 1 LLM call
            final = p_llm if strategy.fastpath != "refine" \
                    else strategy.refine_fn(p_llm, pair, llm)
            return ProvenanceResult(..., fastpath_hit=True)

    # 1–2. Normal BLIP path.
    pruned = strategy.prune_fn(pair, ranker, llm)
    refined = strategy.refine_fn(pruned, pair, llm)
    verified = equivalent(llm.answer(text_of(refined, pair), pair.question),
                          pair.llm_answer)
    assert verified, f"BUG: returned provenance does not verify on pair {pair.pair_id}"
    return ProvenanceResult(..., fastpath_hit=False)
```

The final `assert verified` is intentional. By construction every strategy already verifies its return value — if this assertion ever fires, there's a bug in the strategy, not in the data. The fast-path branch also verifies before returning, so the same invariant holds there.

## 8. Cost ledger

`CostLedger` is a thin container that:

- Accumulates `Usage` events tagged with phase (`"prune"`, `"refine"`, `"verify"`, `"judge"`).
- Computes `cost(usage)` via Eq. 2 using `f_L` per model. `cached_tokens` is the `|Pre|` term; `prompt_tokens - cached_tokens` is the `|P_T \ Pre|` term.
- Exposes `cost_ratio(baseline_cost: float) -> float`. The baseline is `cost(answer(Q, T))` — a single uncached call over the full text, computed once per `(pair)` and stored on the `Pair`.

The ledger does *not* know about strategies — it just records events. The strategy runner attaches strategy metadata when writing the per-pair result row.

## 9. Caching

All disk caches use content-addressed keys (SHA-256 of the canonical inputs). This guarantees: same inputs → same key, regardless of who computed it. Caches:

- Sentence list of a document.
- Block list (depends on sentence list + `m`).
- Token count of a string (depends on encoder + string).
- Embedding of a string (depends on embedding deployment + string).
- `L(T, Q)` answer (depends on driver deployment + `Q` + sentence-list hash + seed).

Caches are append-only and safe under concurrent writers (write to temp file + atomic rename). Cache invalidation is via deletion: bump cache directory name when the cached invariants change.

## 10. Configuration surface

`src/blip/config.py` exposes a single immutable `Config`:

```python
@dataclass(frozen=True)
class Config:
    azure: AzureConfig                       # deployments + key file paths
    cache_dir: Path                          # default: <repo>/cache/
    seed: int = 42
    block_count_m: int = 20                  # Algorithm 1 parameter
    refine_threshold_t: int = 10             # SEQ if pruned size < t else EXP
    f_L_driver: int = 2                      # Eq. 2 cache factor for gpt-54-mini
    f_L_escalation: int = 2
    judge_prompt: str = "llm_equal_human_example"
    judge_model: str = "driver"
    fastpath: str = "refine"                 # "off" | "only" | "refine" | "then_blip"
```

These values mirror the paper's defaults. Anything overrideable from a CLI flag should live here, not scattered through modules.

## 11. What this architecture is *not*

- Not async / not parallel across pairs. The first reproduction milestone runs strategies sequentially per pair. Parallelism is a perf optimization to add later; getting `cost_ratio` correctly accounted is more important than wall-clock.
- Not a service. There is no HTTP layer, no DB. Inputs are JSON files, outputs are JSONL files under `runs/<timestamp>/`.
- Not a framework. Each algorithm is a function, not a class hierarchy. Strategies compose by function composition in `pipeline.py`.
