from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator

from blip._types import Pair, Sentence, Block
from blip.text.segmenter import segment
from blip.text.blocks import build_blocks
from blip.text.tokens import count_tokens

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_PAPERS = _REPO_ROOT / "workload" / "paper" / "query" / "papers.json"
_DEFAULT_SAMPLES = _REPO_ROOT / "workload" / "paper" / "samples"


class PaperWorkload:
    name = "paper"

    def __init__(
        self,
        papers_path: Path | None = None,
        samples_path: Path | None = None,
        seed: int = 42,
    ) -> None:
        self._papers_path = papers_path or _DEFAULT_PAPERS
        self._samples_path = samples_path or _DEFAULT_SAMPLES
        self._seed = seed
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            raw = self._papers_path.read_text(encoding="utf-8")
            assert "�" not in raw, "papers.json contains replacement characters"
            self._data = json.loads(raw)
        return self._data

    def all_qa_pairs(self) -> list[dict]:
        data = self._load()
        pairs = []
        for doi, record in data.items():
            text = record.get("text", "")
            for q in record.get("questions", []):
                pairs.append({
                    "doi": doi,
                    "text": text,
                    "question": q["question"],
                    "ground_truth": q.get("answer"),
                })
        return pairs

    def sample(self, n: int = 20, answerability_filter_fn=None) -> list[dict]:
        """Return up to n (question, doc) pairs, shuffled with fixed seed."""
        pairs = self.all_qa_pairs()
        rng = random.Random(self._seed)
        rng.shuffle(pairs)
        result = []
        for p in pairs:
            if answerability_filter_fn is not None:
                if not answerability_filter_fn(p):
                    continue
            result.append(p)
            if len(result) >= n:
                break
        return result

    def sentences(self, doc_id: str) -> list[str]:
        data = self._load()
        text = data[doc_id]["text"]
        return segment(text)

    def tokens(self, doc_id: str) -> int:
        data = self._load()
        return count_tokens(data[doc_id]["text"])

    def iter_pairs(self) -> Iterator[Pair]:
        """Load from pre-computed sample JSONL if available, else error."""
        sample_file = self._samples_path / f"sample_20_seed{self._seed}.jsonl"
        if not sample_file.exists():
            raise FileNotFoundError(
                f"Sample file not found: {sample_file}. Run precompute first."
            )
        for line in sample_file.read_text().splitlines():
            row = json.loads(line)
            doc_id = row["doi"]
            sents = [
                Sentence(idx=i, text=s, token_count=count_tokens(s))
                for i, s in enumerate(self.sentences(doc_id))
            ]
            blocks = build_blocks(sents)
            yield Pair(
                pair_id=row["pair_id"],
                doc_id=doc_id,
                question=row["question"],
                ground_truth=row.get("ground_truth"),
                llm_answer=row["llm_answer"],
                sentences=tuple(sents),
                blocks=tuple(blocks),
            )
