"""FastAPI app: serves the chat page and streams grounded answers over SSE.

Run:  uvicorn app:app --reload --port 8000   (from the backend/ dir)
Open: http://localhost:8000
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from config import BASE_DIR
import generate
from retrieve import Retriever

app = FastAPI(title="Grounded 10-K QA")

FRONTEND = BASE_DIR / "frontend" / "index.html"

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()  # opens the persisted Chroma collection
    return _retriever


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/ask")
def ask(q: str) -> StreamingResponse:
    def event_stream():
        try:
            for etype, payload in generate.stream(q, get_retriever()):
                yield f"data: {json.dumps({'type': etype, 'payload': payload})}\n\n"
        except Exception as e:  # surface errors to the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'payload': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
