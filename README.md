# Grounded Q&A over six SEC 10-K filings

A small RAG chatbot that answers natural-language questions about six companies'
most recent annual reports - and says **"I don't know"** when the answer isn't
in them. Answers cite the exact source passages they rely on.

> **Positioning:** I optimized for *inspectable* answers, not a pretty UI. Every
> chunk carries its company and filing metadata; every answer must cite retrieved
> passages or refuse. The system is conservative by design - a renter-facing
> agent only works if users can trust what it says, so I'd rather it say "I don't
> know" than invent a number.

---

## The corpus - exact filings used

Most-recent **Form 10-K** per company, pulled from SEC EDGAR and **pinned by exact
accession number** (see `PINNED_ACCESSIONS` in `edgar.py`), so a re-run rebuilds
this exact corpus rather than whatever is newest:

| Company | Ticker | Accession | Filed | Primary doc | Chunks |
|---|---|---|---|---|---|
| Apple Inc. | AAPL | 0000320193-25-000079 | 2025-10-31 | aapl-20250927.htm | 148 |
| JPMorgan Chase & Co. | JPM | 0001628280-26-008131 | 2026-02-13 | jpm-20251231.htm | 1,096 |
| Walmart Inc. | WMT | 0000104169-26-000055 | 2026-03-13 | wmt-20260131.htm | 260 |
| The Coca-Cola Company | KO | 0001628280-26-010047 | 2026-02-20 | ko-20251231.htm | 308 |
| NVIDIA Corporation | NVDA | 0001045810-26-000021 | 2026-02-25 | nvda-20260125.htm | 213 |
| Caterpillar Inc. | CAT | 0000018230-26-000008 | 2026-02-13 | cat-20251231.htm | 259 |

**2,284 chunks total.** Chunk count scales with filing size (JPMorgan's bank 10-K
is ~4x the others) - i.e. chunking is proportional, not over-fragmenting the dense
filings. (Regenerate any time with `python backend/edgar.py` + `python backend/ingest.py`;
the manifest is written to `data/manifest.json`.)

---

## Setup

**Requires Python 3.10+** (developed on 3.11).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Provide your Anthropic API key (the ONLY key required - embeddings are local).
#    Easiest: copy the template and paste your key into .env
cp .env.example .env          # then edit .env -> ANTHROPIC_API_KEY=sk-ant-...
#    (or instead:  export ANTHROPIC_API_KEY=sk-ant-...)
#    optional model override:  ANTHROPIC_MODEL=claude-opus-4-8

# 2. Download the six filings from EDGAR (writes data/manifest.json + data/raw/)
python backend/edgar.py

# 3. Build the vector index (parse -> chunk -> embed -> Chroma at data/chroma/)
python backend/ingest.py

# 4. Run the app
cd backend && uvicorn app:app --port 8000
# open http://localhost:8000

# (optional) run the evaluation
python eval/eval.py
```

Embeddings run locally (Chroma's built-in ONNX MiniLM) - **no embedding API key
needed**, and the index is reproducible offline after the one-time model download.

---

## Decision log

### Architecture
```
EDGAR HTML → table-aware parser → section-tagged chunks (+metadata)
→ local MiniLM embeddings → Chroma (cosine) → retrieval (per-company fan-out
for comparisons) → grounded cite-or-refuse prompt → streamed answer + sources
```
The LLM is the **thin, swappable last layer**. Source passages are produced by
*our* retriever; the model only summarizes and cites what we hand it.

### Chunking
- Parse HTML with BeautifulSoup; strip scripts, styles, and the hidden inline-XBRL
  header. **Tables are flattened to one line per row, keeping the row label with
  its values** (`Net sales | 391,035 | 383,285`) so a single table fact is
  retrievable by embedding. A table is **atomic** - never split mid-row.
- Section-aware, **~200-word chunks** with ~40-word overlap. Size is set by the
  embedder, not guesswork: Chroma's MiniLM truncates at **256 tokens**, so larger
  chunks lose their tail *from the embedding* (the full text is still stored and
  shown to the model). ~200 words ≈ 256 tokens, so the whole chunk is embedded.
  (The plan's "500-800 tokens" predated choosing this lightweight embedder.)
  Section is tracked
  best-effort from `Item N` headers (soft metadata for display/filtering, not the
  retrieval key - so imperfect labels don't hurt grounding).
- Each chunk stores `{company, ticker, filing_date, accession, section, chunk_id}`.

### Embeddings + store
- **Chroma's built-in `all-MiniLM-L6-v2` (ONNX)** - deliberately no
  `sentence-transformers`/torch dependency. Lighter install, reproducible for a
  grader, runs offline. Trade-off: a finance-tuned embedder (e.g. `bge`/Voyage)
  would retrieve better; that's a documented upgrade.
- Cosine space, so `similarity = 1 - distance` feeds the refusal backstop.

### Retrieval
- **Single/unspecified company:** top-k vector search (company-filtered when the
  question names exactly one).
- **Cross-company comparison:** detected **deterministically** (company name/alias
  match + comparative cue words) → retrieve **per-company** and concatenate, so the
  model sees evidence from every relevant filing. This defeats the trap where a
  single merged top-k returns chunks from only one company.
- Deterministic detection is chosen over an LLM classifier for zero latency and
  full inspectability. Known miss: a comparison that names no company and uses no
  cue word - surfaced as a failure case in the eval. Documented upgrade: a small
  LLM query-planner returning `{companies, is_comparison}`.
- Reranking: not implemented (would be the next addition - a cross-encoder over
  the candidate set).

### Grounding & refusal (layered, prompt-first)
1. **Primary - prompt contract:** answer only from the numbered sources, cite
   `[n]` per claim, and reply exactly "I don't know - that isn't in these filings"
   when the specific fact is absent. This is the only mechanism that catches an
   *undisclosed fact about an in-corpus company* (retrieval returns high-similarity
   chunks that don't contain the answer - a similarity threshold would wave it through).
2. **Backstop - relevance floor:** if the best chunk is below `MIN_SIMILARITY`
   (cosine), refuse before calling the model. Catches off-topic / not-in-corpus
   questions (e.g. "Tesla's revenue?"). The floor is tuned against the eval set
   and is a backstop, not the primary mechanism.

### Refusal calibration (how the threshold was set)
The prompt contract is the effective refusal mechanism - the eval confirms it
(4/4 out-of-corpus + undisclosed questions refused correctly, **0 false
refusals**). The similarity backstop (`MIN_SIMILARITY = 0.20`) is deliberately
left low: valid lookups score 0.55-0.80 and off-topic queries (e.g. "Tesla")
score ~0.43, so a higher floor (~0.48) would add defense-in-depth but risks
false-refusing legitimate low-similarity comparison queries. I chose to let the
prompt do the refusing and keep the backstop as a dormant safety net for
degenerate (near-zero-similarity) inputs.

### Where it currently breaks / is weak
- **Retrieving a specific number from a dense table is the core weakness (both eval
  failures, q06 and q09).** When a figure lives only in a financial-statement table
  with little surrounding prose, it ranks low: Coca-Cola's net operating revenues
  ($47,941M) sits at rank ~20, and Walmart's total revenues ($713,163M) isn't in
  Walmart's own top-20 for "highest total revenue" - so the cross-company comparison
  can't see it. Apple's total net sales had the same shape but at rank 8, so I
  widened `TOP_K` to 12 - the point where legitimate split-table facts surface
  without diluting answers or causing false refusals (verified against the eval:
  refusals stayed 4/4). I deliberately did **not** keep raising it to force
  Coca-Cola/Walmart through - those stay documented failures rather than a number
  tuned until the test passed. The companies that *do* retrieve cleanly
  (Caterpillar, Apple, JPMorgan) restate the figure in strongly-matching MD&A prose.
  Real fix: **hybrid (keyword + vector) retrieval**, or keeping each statement's
  header row with its value rows.
- **Table rows can lose their column header** (which fiscal year a number belongs
  to) when a multi-row statement spans chunks.
- **Comparison detection is deterministic** - a comparison naming no company and
  using no cue word stays in single-doc "open" mode (latent in q15, which happened
  to surface multiple companies anyway).
- **Section labels are approximate** - `Item N` headers are detected heuristically;
  financial statements sometimes get tagged with a neighboring item number.
- **Embedder is general-purpose MiniLM** with a 256-token window (drove the
  ~200-word chunk size).

### What I'd do with another week
- **Hybrid retrieval** (BM25/keyword + vector) - directly fixes q06-type failures
  where the answer is a specific number with little surrounding prose.
- Finance-tuned embeddings (`bge`/Voyage) + a cross-encoder reranker.
- LLM query-planner for entity/intent (covers paraphrased comparisons).
- Better table structure (carry multi-level headers + statement caption into every
  value chunk).
- Numeric-aware eval (parse figures and check tolerance, not just substring match).

---

## Evaluation

`eval/qa_pairs.yaml` holds hand-written Q/A pairs spanning single-fact prose,
table reads, cross-company comparisons, and out-of-corpus refusals. Run:

```bash
python eval/eval.py     # reads ANTHROPIC_API_KEY from .env
```

### Results: 13 / 15 correctly grounded

| Category | Questions | Pass |
|---|---|---|
| Single-fact (prose) | q01-q03 | 3/3 |
| Single-fact (**table read**) | q04-q08 | 4/5 |
| Cross-company comparison | q09-q10 | 1/2 |
| Out-of-corpus refusal | q11-q13 | 3/3 |
| Undisclosed in-corpus refusal | q14 | 1/1 |
| Comparison-detection stress | q15 | 1/1 |

What works:
- **Table reads:** NVIDIA revenue ($215,938M), Apple net sales ($416,161M),
  Apple R&D ($34,550M), JPMorgan net revenue ($185,581M) all retrieved and cited.
  This only works because of contextual chunking (prepending company + section to
  each chunk) - without it, number-dense chunks ranked ~20th behind boilerplate.
  Measured: NVIDIA revenue went from retrieval rank 19 → rank 1.
- **Honesty:** every out-of-corpus question (Tesla, Microsoft, Amazon) and the
  undisclosed-in-corpus question (Apple churn rate) refused correctly - no invented
  numbers.
- **One comparison works:** "highest R&D" → Apple ($34,550M), correctly above
  Caterpillar's $2,148M, with an honest caveat that NVIDIA only disclosed R&D as a
  percentage in the retrieved chunks. The per-company fan-out put every company's
  evidence in front of the model.

The two failures - **same root cause**, the most honest finding in this project:
- **q06 (Coca-Cola net operating revenues):** the value sits at retrieval rank ~20;
  the income-statement label and its numbers are in separate chunks and KO doesn't
  restate the figure in matching prose. The system **refused rather than guessed** -
  correct-by-design even when it's a miss.
- **q09 (highest total revenue):** the fan-out couldn't retrieve Walmart's
  total-revenue chunk ($713,163M is not even in Walmart's own top-20 for this query),
  so the model compared only the companies whose figures it *could* see. It answered
  honestly ("based on the available sources…") instead of fabricating a ranking.

Both are the **same weakness**: retrieving a *specific number* that lives only in a
dense table with little surrounding prose. The fix is hybrid (keyword + vector)
retrieval - top of the "another week" list.

> **A note on what "correct" means.** q09 originally *false-passed*: my first
> scoring only checked for the substring "Walmart", which appeared even though the
> answer said Walmart's figures were blank. I tightened the check to require the
> actual figure ($713,163) - which exposed the real failure and dropped the score
> from a flattering 14/15 to an honest 13/15. Substring scoring is also unit-fragile
> (a correct "$67.6B" wouldn't match a "$67,589M" token); numeric-tolerant scoring
> is on the "another week" list. The eval is a *measurement aid*, not the source of
> truth - I read every answer by hand.
