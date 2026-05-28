# BLIP — Bolt-on, Verifiable Provenance for LLM-Powered Data Processing

BLIP finds a **minimal text provenance** for any LLM answer: given a long document and a question, it returns the shortest subsequence of sentences such that re-running the LLM on only those sentences produces an equivalent answer. This guarantees the answer is grounded in the text with zero false positives (100% verifiability).

The pipeline runs in two phases — **Prune** (discard irrelevant blocks) then **Refine** (trim to a minimal subsequence) — and treats the LLM as a black box.

---

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies:** `openai>=1.30`, `pysbd>=0.3.4`, `tiktoken>=0.7`, `numpy>=1.26`

---

## Setup

Create `local/azure.json` with your Azure OpenAI deployments:

```json
{
  "key_file": "path/to/gpt4o.txt",
  "key_file_cheap": "path/to/gpt4o-mini.txt",
  "embedding_key_file": "path/to/text-embedding-3-small.txt"
}
```

Each key file contains your Azure endpoint and API key. The `cheap` deployment is used as the driver model for most LLM calls; the main deployment is used for escalation.

---

## Usage

### Step 1 — Precompute baseline answers

Sample Q-A pairs from the workload and cache the full-document LLM answers. This must be run once before any experiments.

```bash
python -m blip.runner.precompute --n 20 --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 20 | Number of Q-A pairs to sample |
| `--seed` | 42 | Random seed for reproducibility |
| `--max-docs` | 500 | Maximum number of source documents to draw from |
| `--smoke-test` | off | Run on a single pair for a quick sanity check |

Output: `workload/paper/samples/sample_N_seedS.jsonl`

### Step 2 — Run a provenance strategy

```bash
python -m blip.runner.experiment --strategy STRATEGY [--n 20] [--seed 42]
```

Results are written to `runs/{strategy}/{n}pairs_{timestamp}/results.jsonl`.

#### Strategy name format

```
{ranker}_{scan}_{refine}[_{fastpath}]
```

| Component | Options | Description |
|-----------|---------|-------------|
| `ranker` | `embedding`, `llm` | How blocks are ranked in the prune phase |
| `scan` | `bottom_up`, `top_down`, `adaptive` | Order blocks are tested during pruning |
| `refine` | `none`, `seq`, `exp`, `auto` | Refinement algorithm after pruning |
| `fastpath` | `off`, `only`, `refine`, `then_blip` | LLM-citation front door (optional) |

**Example strategies:**

```bash
# Embedding ranker, adaptive prune, exponential refine (recommended)
python -m blip.runner.experiment --strategy embedding_adaptive_exp

# LLM ranker, top-down prune, no refine
python -m blip.runner.experiment --strategy llm_top_down_none

# Fast-path only (LLM self-citation baseline)
python -m blip.runner.experiment --strategy embedding_bottom_up_seq_only
```

---

## Output format

Each row in `results.jsonl` contains:

```json
{
  "pair_id": "...",
  "verified": true,
  "size_ratio": 0.08,
  "cost_ratio": 1.05,
  "stage1_cost_usd": 0.0003,
  "stage2_cost_usd": 0.0001,
  "stage1_latency_s": 1.2,
  "stage2_latency_s": 0.4,
  "provenance_sentences": ["...", "..."]
}
```

- `verified` — whether `I(L(P,Q), A) = True` (provenance answer matches baseline)
- `size_ratio` — provenance token length / full document token length
- `cost_ratio` — total cost / cost of one full-document call

---

## Programmatic API

```python
from blip.config import load_config
from blip.llm.client import LLMClient
from blip.rank.embedding import EmbeddingRanker
from blip.pipeline import StrategySpec, run
from blip.cache.disk import DiskCache

cfg = load_config()
llm = LLMClient(cfg)
cache = DiskCache(cfg.cache_dir)
ranker = EmbeddingRanker(llm, cache=cache)

strategy = StrategySpec(
    name="embedding_adaptive_exp",
    ranker="embedding",
    scan="adaptive",
    refine="exp",
    fastpath="refine",
)

result = run(pair, strategy, ranker, llm)
print(result.provenance_idxs)   # sentence indices of minimal provenance
print(result.verified)          # True if equivalence check passed
print(result.cost_ratio)        # cost relative to full-document answering
```

---

## Running tests

```bash
pytest
```

---

## Project structure

```
src/blip/
├── pipeline.py        # top-level run() entry point
├── config.py          # load Azure keys and hyperparameters
├── _types.py          # core dataclasses (Sentence, Block, Pair, ProvenanceResult)
├── algo/
│   ├── prune.py       # Algorithms 1: bottom-up, top-down, adaptive prune
│   ├── refine.py      # Algorithm 2: sequential and exponential greedy refine
│   ├── fastpath.py    # LLM-citation front door
│   └── topk.py        # Algorithm 3: top-k provenance
├── rank/
│   ├── embedding.py   # cosine-similarity block ranker
│   └── llm.py         # LLM-scored block ranker
├── llm/
│   ├── client.py      # Azure OpenAI wrapper with retry and KV-cache tracking
│   ├── judge.py       # equivalence checker I(A, A')
│   └── prompts.py     # all prompt templates
├── cost/              # cost accounting with KV-cache discount model
├── cache/             # SQLite-based disk memoization
├── text/              # sentence segmentation and token counting
├── workloads/         # dataset adapters (paper workload: 584 papers, 7961 Q-A pairs)
└── runner/
    ├── precompute.py  # offline baseline answer caching
    └── experiment.py  # strategy execution and result logging
```
