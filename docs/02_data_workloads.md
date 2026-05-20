# Data & Workloads

This reproduction uses one workload: the local `paper` corpus under `workload/paper/`. Future workloads will plug into the same adapter interface described in §4.

## 1. What's in `workload/paper/`

```
workload/paper/
├── data/
│   └── raw_data/                       # 102 PDFs (subset of the JSON records)
│       ├── A Lived Informatics Model of Personal Informatics.pdf
│       ├── …
│       └── Wearing Uncomfortable …pdf
└── query/
    └── papers.json                     # 584 paper records with text + questions
```

Counts:

| Item                  | Count |
| --------------------- | ----- |
| Paper records         | 584   |
| PDFs on disk          | 102   |
| Total questions       | 7,961 |
| Avg questions / paper | 13.63 |
| Min / max questions   | 10 / 17 |

The mismatch (584 records vs. 102 PDFs) means the PDFs are a *subset*. Since the JSON already contains pre-extracted `text`, the PDFs are not needed for the BLIP pipeline — they are useful only for spot-checking that the JSON `text` is faithful.

## 2. Record schema

Each entry in `papers.json` is keyed by DOI URL:

```json
{
  "https://doi.org/10.1145/3335082.3335100": {
    "title": "\"I Don't Understand...\" Issues in Self-Quantifying Commuting",
    "text": "<full extracted text, ~25,000 chars in this example>",
    "questions": [
      {"question": "Where was this paper published?", "answer": "ECCE '19"},
      {"question": "In what year was this paper published?", "answer": "2019"},
      {"question": "Who are the authors of this paper?", "answer": "C. Boulard, S. Castellani, …"},
      …
    ]
  },
  …
}
```

Notes for the implementer:

- `text` is OCR-style extracted text from the PDF; expect occasional typos (e.g., "1 Don't Understand" instead of "I Don't Understand"). BLIP does not depend on perfect text — it only requires the LLM to be able to answer over `text`.
- `questions` is a mix of question categories. From a quick scan, the categories overlap with the paper's taxonomy (Table 5): **Look-up** (year, venue, authors), **Aggregation** (counts, lists of keywords), **Judge** (yes/no — "Does this paper involve expert participants?"), and a small amount of **Reasoning**.
- `answer` is the *ground-truth* answer provided by the dataset curators. BLIP does **not** use this as the ground truth for provenance — BLIP uses `L(T, Q)`, the LLM's own answer on the full text, as the target. The ground-truth `answer` is useful only for (a) sanity-checking that `gpt-4o-mini` is competent on this corpus and (b) optional accuracy reporting.

## 3. Sampling protocol

The paper samples 500 distinct (question, document) pairs per workload (§5.1). We adopt that count. The block size `m = 20` is also from the paper; the sentence segmenter (`pysbd`) and tokenizer (`tiktoken o200k_base`) are reproduction choices and **not** paper-specified — they are our defaults, swappable as long as round-tripping holds.

1. Load all 584 papers, all 7,961 questions.
2. Shuffle with fixed seed (e.g., `seed=42`).
3. Take the first 500 (question, paper) pairs that pass the *answerability filter*: `L(T_i, Q_j)` returns a non-trivial answer (i.e., not "I cannot find …"). Skipped pairs are logged.
4. Persist the 500-pair sample to `workload/paper/samples/sample_500_seed42.jsonl` so the same evaluation set is reused across strategies and across runs.

Each sample row:

```json
{
  "pair_id": "0001",
  "doi": "https://doi.org/10.1145/…",
  "question": "What domains does this paper address?",
  "ground_truth": "sustainability",
  "llm_answer": "sustainability",          // L(T, Q), computed once and cached
  "llm_answer_tokens": 1,
  "text_tokens": 5832,
  "text_sentence_count": 412
}
```

`llm_answer` is the *target* `A` used by every strategy. Caching it is critical: BLIP's verification step compares `L(P, Q)` to this `A`, so `A` must be deterministic for the run.

## 4. Workload-adapter interface

To make adding other workloads (NL_DEV, HotpotQA, …) painless later, the loader exposes a uniform shape:

```python
class WorkloadAdapter(Protocol):
    name: str                              # e.g., "paper"
    def iter_pairs(self) -> Iterator[Pair]: ...
    def sentences(self, doc_id: str) -> list[str]: ...
    def tokens(self, doc_id: str) -> int: ...
```

`Pair` carries `doc_id`, `question`, `ground_truth`, and the cached `llm_answer`. The `paper` adapter is implemented in Milestone 0 (see `06_roadmap.md`); subsequent adapters follow the same protocol.

## 5. Sentence segmentation

BLIP operates at the sentence level: every algorithm reasons in terms of "remove sentence `s_i`", "rank block `b_j`", etc. We use a single consistent segmenter:

- **Tool:** `pysbd` (Python Sentence Boundary Disambiguation). Robust to abbreviations ("e.g.", "Dr.", "Fig.") and to scientific text. spaCy is acceptable as a fallback but heavier.
- **Pre-processing:** strip page headers/footers if regex-detectable (e.g., "Page 3 of 18"); normalize whitespace; do not lowercase.
- **Idempotence:** segmenting and re-joining with single spaces must round-trip without losing characters. We hash and check this on every load.

The segmented sentence list is the canonical representation of `T` for the rest of the pipeline.

## 6. Blocks vs. sentences

Algorithm 1 (Prune) operates on **blocks** `B = ⟨b_1, …, b_m⟩` where `m = 20` (paper default; balances ranker cost against pruning granularity). Each block is a contiguous run of sentences such that block sizes differ by at most one.

Algorithm 2 (Refine) operates on **sentences** directly.

So the pipeline keeps two parallel representations of `T`:

- `T_sentences = [s_1, …, s_n]`
- `T_blocks = [b_1, …, b_20]` where each `b_j` is `T_sentences[a_j:a_{j+1}]`

Both are derived from the same segmenter output to guarantee `concat(T_blocks) == T_sentences`.

## 7. Token counting

- **Tokenizer:** `tiktoken` with the `o200k_base` encoder (the encoding used by the `gpt-4o` / `gpt-4o-mini` family that backs the Azure `gpt-54-mini` deployment). Token counts feed (a) the cost model and (b) the `text_tokens` field above.
- We count tokens once per document and cache to disk, since tokenization is hot in the cost-ratio computation.

## 8. Offline-precomputed assets

For each paper in the sampled set, before any experiment runs, we precompute and cache:

| Asset                   | Location                                     | Used by              |
| ----------------------- | -------------------------------------------- | -------------------- |
| Sentence list           | `cache/sentences/<doi_hash>.json`            | All strategies       |
| Block list (m=20)       | `cache/blocks/<doi_hash>.json`               | Algorithm 1          |
| Token count of text     | `cache/tokens/<doi_hash>.txt`                | Cost ratio           |
| Embedding per sentence  | `cache/embeddings/<doi_hash>.npy`            | Embedding ranker     |
| Embedding per block     | `cache/embeddings/<doi_hash>_blocks.npy`     | Embedding ranker     |
| Question embedding      | `cache/embeddings/q_<q_hash>.npy`            | Embedding ranker     |
| `L(T, Q)` per pair      | `cache/answers/<doi_hash>_<q_hash>.json`     | Verification target  |

The first run populates the cache; subsequent runs only do LLM calls for prune/refine/verify.

## 9. What is intentionally *not* in this workload

- No SQL/tabular ground-truth tuples (so no TQA-bench `R` / `eR` evaluation).
- No multi-document questions (each question is anchored to one paper).
- No human-curated provenance, so we cannot report true-provenance recovery — only verifiability, size, cost, and latency.

These limitations are acceptable for the first reproduction milestone. They constrain the experiments to those listed in `05_evaluation_plan.md` §3.
