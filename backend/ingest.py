"""Build the vector index: manifest -> parse + chunk -> embed -> Chroma.

Run:  python backend/ingest.py
Assumes filings are already downloaded (backend/edgar.py wrote data/manifest.json).
Embeddings use Chroma's built-in ONNX all-MiniLM-L6-v2 (no torch dependency);
cosine space so similarity = 1 - distance for the refusal backstop.
"""
from __future__ import annotations

import json
import shutil

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, MANIFEST_PATH
from parse import chunk_filing


def build(reset: bool = True) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())   # the 6 filings edgar.py downloaded
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)                       # wipe old index -> clean rebuild
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))   # open Chroma on disk
    col = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}     # cosine -> similarity = 1 - distance
    )

    total = 0
    for entry in manifest:                              # one filing at a time
        html = open(entry["local_path"], encoding="utf-8", errors="ignore").read()  # read its HTML
        chunks = chunk_filing(html, entry)              # cut into table-aware, labeled chunks (parse.py)
        # add in batches to stay well under any payload limits
        for i in range(0, len(chunks), 256):
            batch = chunks[i : i + 256]
            col.add(                                                # <-- embedding happens here (MiniLM, automatic)
                ids=[c["id"] for c in batch],                       # "NVDA-0007", ...
                documents=[c["text"] for c in batch],               # the chunk text (gets embedded)
                metadatas=[c["metadata"] for c in batch],           # company/ticker/section/etc.
            )
        total += len(chunks)
        print(f"  {entry['ticker']:5} {entry['company']:28} -> {len(chunks):4} chunks")

    print(f"\nindexed {total} chunks across {len(manifest)} filings into '{COLLECTION_NAME}'")
    print(f"chroma persisted at {CHROMA_DIR}")


if __name__ == "__main__":
    build()
