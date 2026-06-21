"""Central config: paths, model id (env-driven), retrieval + refusal knobs.

Everything tunable lives here so the walkthrough has one place to point at,
and the LLM choice is a one-line, swappable change (the "thin last layer").
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Quiet the HF tokenizers fork warning emitted by Chroma's embedder.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- paths ----
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env at the project root (if present) so ANTHROPIC_API_KEY / ANTHROPIC_MODEL
# can live in a file instead of being exported each session.
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "tenk_chunks"

# ---- generation (swappable; verify console access on first run) ----
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 1024

# ---- chunking ----
# Chroma's ONNX all-MiniLM-L6-v2 truncates at 256 tokens (verified:
# tokenizer.enable_truncation(max_length=256)). Chunks larger than that have
# their tail dropped from the *embedding* (full text is still stored/shown to
# the model). ~200 words ≈ ~256 tokens, so the whole chunk is embedded.
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40   # carry context across boundaries

# ---- retrieval ----
TOP_K = 12                 # single-company / default question (income-statement
                           # totals can rank ~8-10, so give the model headroom)
PER_COMPANY_K = 3          # chunks per company in a cross-company fan-out
                           # (single source of truth: used for both the
                           # per-company retrieval depth and the context cap)

# ---- refusal backstop ----
# Chroma is configured for cosine distance, so similarity = 1 - distance.
# This floor is a BACKSTOP for off-topic / not-in-corpus questions; the prompt
# contract is the primary refusal mechanism. Tuned against the eval set.
MIN_SIMILARITY = 0.20

REFUSAL_TEXT = "I don't know - that isn't in these filings."
