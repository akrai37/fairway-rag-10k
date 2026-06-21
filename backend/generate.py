"""Grounded generation: turn retrieved chunks into a cited answer or a refusal.

Refusal is layered and prompt-first:
  1. PRIMARY  - the system prompt forbids using anything outside the numbered
     sources and requires an explicit "I don't know" when the specific fact is
     absent. This is the only thing that catches an undisclosed fact about an
     in-corpus company (high-similarity chunks that don't contain the answer).
  2. BACKSTOP - if the best retrieved chunk is below MIN_SIMILARITY we refuse
     before calling the model (off-topic / not-in-corpus, e.g. "Tesla").

Citations are [n] references to OUR retrieved chunks - grounding is owned by
the pipeline, not a model feature.
"""
from __future__ import annotations

import os

import anthropic

from config import ANTHROPIC_MODEL, MAX_TOKENS, MIN_SIMILARITY, PER_COMPANY_K, REFUSAL_TEXT
from retrieve import Hit, Retriever, Retrieval

SOURCE_CAP = 14  # max chunks fed as context (keeps tokens bounded on fan-out)

SYSTEM = (
    "You are a careful assistant answering questions about six companies' most "
    "recent SEC Form 10-K filings (Apple, JPMorgan Chase, Walmart, Coca-Cola, "
    "NVIDIA, Caterpillar). You are given numbered source passages retrieved from "
    "those filings.\n\n"
    "Rules:\n"
    "1. Use ONLY facts found in the numbered sources. Never use outside knowledge.\n"
    "2. Cite the source number(s) like [1], [3] after every factual claim.\n"
    "3. If the specific fact needed to answer is NOT in the sources, reply exactly: "
    f'"{REFUSAL_TEXT}" You may add one short sentence on what is missing. Do not '
    "guess, estimate, or fill gaps from general knowledge.\n"
    "4. For comparisons, only compare companies whose figures actually appear in "
    "the sources; if a company's figure is missing, say so rather than inferring.\n"
    "5. Report figures with the units and period exactly as stated in the source.\n"
    "6. Be concise."
)


def select_sources(retr: Retrieval) -> list[Hit]:
    # For comparisons, keep the top-N chunks PER company (preserving global
    # similarity order) so a low-scoring-but-relevant company isn't dropped by a
    # global cap. For single/open queries, a plain top-k is fine.
    if retr.mode == "fan-out":
        per: dict[str, int] = {}
        kept: list[Hit] = []
        for h in retr.hits:  # already sorted by similarity               
            t = h.metadata["ticker"]
            if per.get(t, 0) < PER_COMPANY_K:
                per[t] = per.get(t, 0) + 1
                kept.append(h)
        return kept
    return retr.hits[:SOURCE_CAP]


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.metadata
        blocks.append(
            f"[{i}] ({m['company']}, {m['section']}, filed {m['filing_date']})\n{h.text}"
        )
    return "\n\n".join(blocks)


def sources_payload(hits: list[Hit]) -> list[dict]:
    out = []
    for i, h in enumerate(hits, 1):
        m = h.metadata
        out.append({
            "n": i,
            "company": m["company"],                                             
            "ticker": m["ticker"],
            "section": m["section"],
            "accession": m["accession"],
            "filing_date": m["filing_date"],
            "similarity": round(h.similarity, 3),
            "text": h.text,
        })
    return out


def build_messages(query: str, hits: list[Hit]) -> list[dict]:
    user = (
        f"Question: {query}\n\n"
        f"Numbered source passages:\n\n{format_context(hits)}\n\n"
        "Answer using only these sources, with [n] citations, or say you don't know."
    )
    return [{"role": "user", "content": user}]


def get_client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    return anthropic.Anthropic()


def _early_refuse(hits: list[Hit]) -> bool:
    """Backstop: nothing retrieved clears the relevance floor."""
    return not hits or hits[0].similarity < MIN_SIMILARITY


def answer(query: str, retriever: Retriever, client: anthropic.Anthropic | None = None) -> dict:
    """Non-streaming answer (used by the eval harness)."""
    retr = retriever.retrieve(query)
    hits = select_sources(retr)
    meta = {"mode": retr.mode, "companies": retr.companies, "comparison": retr.comparison}

    if _early_refuse(hits):
        return {"answer": REFUSAL_TEXT, "sources": [], "refused": True,
                "refusal_stage": "backstop", **meta}

    client = client or get_client()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM,
        messages=build_messages(query, hits),
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    refused = text.startswith(REFUSAL_TEXT[:20])
    return {
        "answer": text,
        "sources": [] if refused else sources_payload(hits),
        "refused": refused,
        "refusal_stage": "prompt" if refused else None,
        **meta,
    }


def stream(query: str, retriever: Retriever, client: anthropic.Anthropic | None = None):
    """Yield events for SSE: ('meta'|'token'|'sources'|'done', payload)."""
    retr = retriever.retrieve(query)
    hits = select_sources(retr)
    yield ("meta", {"mode": retr.mode, "companies": retr.companies,
                    "comparison": retr.comparison})

    if _early_refuse(hits):
        yield ("token", REFUSAL_TEXT)
        yield ("sources", [])
        yield ("done", {"refused": True, "refusal_stage": "backstop"})
        return

    client = client or get_client()
    collected = ""
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM,
        messages=build_messages(query, hits),
    ) as s:
        for delta in s.text_stream:
            collected += delta
            yield ("token", delta)

    refused = collected.strip().startswith(REFUSAL_TEXT[:20])
    yield ("sources", [] if refused else sources_payload(hits))
    yield ("done", {"refused": refused, "refusal_stage": "prompt" if refused else None})   #if refused is True → "prompt" (meaning the AI itself refused)


if __name__ == "__main__":
    import sys
    r = Retriever()
    q = " ".join(sys.argv[1:]) or "What was Apple's total net sales?"
    out = answer(q, r)
    print(f"Q: {q}\n[{out['mode']}] refused={out['refused']} ({out['refusal_stage']})\n")
    print(out["answer"])
    if out["sources"]:
        print("\nSources:")
        for s in out["sources"]:
            print(f"  [{s['n']}] {s['ticker']} {s['section'][:28]:28} sim={s['similarity']}")
