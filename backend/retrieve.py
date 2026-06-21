"""Retrieval over the Chroma index.

Two behaviours, chosen by deterministic, inspectable query analysis:
- Single / unspecified company -> ordinary top-k (optionally company-filtered).
- Cross-company comparison    -> per-company fan-out so the model sees evidence
                                 from every relevant filing, defeating the
                                 "can only retrieve from one document" trap.

Company + comparison detection is rule-based on purpose: zero latency, fully
explainable, and easy to point at in the walkthrough. The documented upgrade is
a small LLM query-planner; the known miss (a comparison that names no company
and uses no cue word) is surfaced as a failure case in the eval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K, PER_COMPANY_K

# alias -> ticker. Names are matched as substrings; bare tickers need word
# boundaries so "cat" doesn't fire inside "category".
_COMPANY_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM", "jpm": "JPM",
    "walmart": "WMT", "wal-mart": "WMT", "wmt": "WMT",
    "coca-cola": "KO", "coca cola": "KO", "coke": "KO",
    "nvidia": "NVDA", "nvda": "NVDA",
    "caterpillar": "CAT",
}
_BARE_TICKERS = {"AAPL", "JPM", "WMT", "KO", "NVDA", "CAT"}
ALL_TICKERS = ["AAPL", "JPM", "WMT", "KO", "NVDA", "CAT"]

# Comparison cues. Kept specific to avoid false positives: bare "most"/"least"
# would fire on "most recent", "between" on date ranges, "across the" on
# "across the company". Real two-company comparisons are already caught by
# detecting 2+ company names, so we don't need those loose cues here.
_COMPARE_CUES = [
    "compare", "comparison", "versus", " vs ", "which of", "which company",
    "which one", "highest", "lowest", "largest", "smallest", "greatest",
    "biggest", "rank ", "more than", "less than", "all six",
    "each company", "across these companies",
]


@dataclass
class Hit:
    text: str
    metadata: dict
    similarity: float


def detect_companies(query: str) -> list[str]:
    q = query.lower()
    found: list[str] = []
    for alias, ticker in _COMPANY_ALIASES.items():
        if alias in q and ticker not in found:
            found.append(ticker)
    # bare tickers with word boundaries (e.g. "CAT", "KO")
    for ticker in _BARE_TICKERS:
        if ticker not in found and re.search(rf"\b{ticker.lower()}\b", q):
            found.append(ticker)
    return found


def is_comparison(query: str) -> bool:
    q = f" {query.lower()} "
    return any(cue in q for cue in _COMPARE_CUES)


@dataclass
class Retrieval:
    hits: list[Hit]
    mode: str               # "fan-out" | "single-company" | "open"
    companies: list[str]
    comparison: bool


class Retriever:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection(COLLECTION_NAME)

    def _query(self, query: str, k: int, ticker: str | None) -> list[Hit]:
        where = {"ticker": ticker} if ticker else None
        res = self.col.query(query_texts=[query], n_results=k, where=where)
        hits: list[Hit] = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(Hit(text=doc, metadata=meta, similarity=1.0 - dist))
        return hits

    def retrieve(self, query: str) -> Retrieval:
        companies = detect_companies(query)
        comparison = is_comparison(query)

        if comparison or len(companies) >= 2:
            targets = companies if len(companies) >= 2 else ALL_TICKERS
            hits: list[Hit] = []
            for t in targets:
                hits.extend(self._query(query, PER_COMPANY_K, t))
            mode = "fan-out"
        elif len(companies) == 1:
            hits = self._query(query, TOP_K, companies[0])
            mode = "single-company"
        else:
            hits = self._query(query, TOP_K, None)
            mode = "open"

        hits.sort(key=lambda h: h.similarity, reverse=True)
        return Retrieval(hits=hits, mode=mode, companies=companies, comparison=comparison)


if __name__ == "__main__":
    import sys
    r = Retriever()
    q = " ".join(sys.argv[1:]) or "What was Apple's total net sales?"
    out = r.retrieve(q)
    print(f"Q: {q}")
    print(f"mode={out.mode} companies={out.companies} comparison={out.comparison}")
    for h in out.hits[:6]:
        print(f"  [{h.similarity:.3f}] {h.metadata['ticker']} {h.metadata['section'][:30]:30} {h.text[:90]}")
