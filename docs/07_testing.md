# Testing Plan

The single test that matters: **does the returned provenance reproduce the answer?**

Everything else is optional.

## 1. The core test

For each (text `T`, question `Q`) pair:

1. Compute `A = L(T, Q)` once and cache it.
2. Run BLIP → get provenance `P`.
3. Compute `A' = L(P, Q)`.
4. **Pass** iff `I(A, A') == True` (the equivalence judge).

Pseudocode:

```
def test_reproduces(pair, strategy):
    A      = pair.llm_answer                     # cached
    P      = strategy.run(pair)
    A_pri  = llm.answer(text_of(P), pair.question)
    assert judge.equivalent(A, A_pri), f"{pair.pair_id}: P did not reproduce A"
```

That's it. If `A'` matches `A`, the result is good. We do not separately check minimality, sortedness, cache hit rates, or any other property in tests. Those are *invariants the algorithm should maintain*, but the only thing the user cares about is "does the small provenance reproduce the answer".

## 2. How we apply it

Two scales:

- **Smoke (5–10 pairs, before any full run).** Pick 10 pairs from the sample, run the chosen strategy, assert every one reproduces. Quick gate — catches gross bugs (wrong text passed in, judge always returning False, etc.) before spending money on the full 500.
- **Full (500 pairs, the experiment itself).** Every row in `runs/<ts>/<strategy>.jsonl` already records `verified: true/false` (the assertion inside `pipeline.run`). Aggregator script reads the JSONL and reports the pass rate. We expect 1.0; anything less is a bug to fix, not a number to report.

Both use the same check from §1.

## 3. Smoke fixtures (optional, for development without Azure)

A handful of tiny hand-written documents to develop against when offline:

| Fixture       | Doc                                              | Question                | Answer | Provenance |
| ------------- | ------------------------------------------------ | ----------------------- | ------ | ---------- |
| `lookup_year` | 12-sentence study description; s_8: "…in 2021." | "What year?"            | "2021" | `{s_8}`    |
| `yes_no`      | s_4: "All participants were experts."           | "Are participants experts?" | "Yes"  | `{s_4}`    |
| `multi_hop`   | s_2 introduces a fact, s_9 refines it             | "Where is the funding from?" | (str) | `{s_2, s_9}` |

For each fixture: a fake LLM whose answer is a deterministic function of which sentences are in `P` (so we know the right provenance up front). Run BLIP under this fake → check `P` reproduces. This lets us iterate on the algorithm without burning Azure credits.

These fixtures are convenience, not requirements. The core check in §1 is what gates a real run.

## 4. Two plumbing checks (one-time)

Before the first real run, two quick health checks:

1. **Azure smoke.** One real chat call to `gpt-54-mini`. Returns text, returns a non-zero `usage.prompt_tokens`. Confirms the key file, deployment name, and SDK version are correct.
2. **Judge sanity.** Five hand-picked equivalent and five non-equivalent answer pairs (e.g., `"2019"`/`"2019"`, `"Maya, an officer"`/`"officer Maya"`, `"yes"`/`"no"`). Run the judge. Expect all 10 correct.

If either fails, fix it before running the experiment. Neither is a recurring test.

## 5. What I deliberately dropped from the earlier version of this doc

- Unit tests per module
- Property-based tests
- Mutation tests
- KV-cache discipline checks
- Sortedness / minimality / determinism property tests
- CI test layout

All of those were good engineering hygiene, but they tested *how* BLIP works rather than *whether* it works. Since the verifiability check in §1 already catches every failure that matters to the user, the additional layers were over-engineering.

If a specific bug bites us during development that the §1 check doesn't catch, we add a targeted regression test then — not before.
