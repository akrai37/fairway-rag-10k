"""Parse a 10-K HTML filing into clean, section-tagged, table-aware chunks.

Design choices (defended in the decision log):
- Tables are flattened to one line per row, label kept with its values
  ("Net sales | 391,035 | 383,285"), so a single-fact table lookup is
  retrievable by embedding. A table is treated as an ATOMIC block - never
  split mid-row; only an oversized table is split on row boundaries.
- Section is tracked best-effort from "Item N" headers. It's soft metadata
  for display/filtering, not the retrieval key, so imperfect labels are fine.
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, Tag
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from config import CHUNK_WORDS, CHUNK_OVERLAP_WORDS

# "Item 1A.", "Item 7.", "ITEM 8" ... - header if the line is short (not a TOC row with page numbers)
_ITEM_RE = re.compile(r"^\s*item\s+(\d+[a-z]?)\s*[.\:\-]?\s*(.*)$", re.IGNORECASE)
_PAGENUM_RE = re.compile(r"^\d{1,4}$")


def _is_boilerplate(line: str) -> bool:
    """Repetitive filing noise that crowds out real chunks in retrieval."""
    low = line.lower()
    return (
        low.startswith("see accompanying notes")
        or low == "table of contents"
        or bool(_PAGENUM_RE.match(line))
    )
_WS_RE = re.compile(r"[ \t ]+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def _flatten_table(table: Tag) -> str:
    """Render a table as one line per non-empty row: 'label | v1 | v2'."""
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", "(", ")"}]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline-XBRL header is huge and display:none - drop it and any hidden node.
    # Guard against nodes already decomposed as a descendant (attrs becomes None).
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        name = (tag.name or "").lower()
        if name.startswith("ix:"):
            tag.decompose()
            continue
        style = (tag.get("style") or "").replace(" ", "").lower()
        if "display:none" in style:
            tag.decompose()


def _segments(html: str) -> list[tuple[str, str, bool]]:
    """Linear (section, text, is_table) segments in document order."""
    soup = BeautifulSoup(html, "lxml")
    _strip_noise(soup)
    body = soup.body or soup

    # Replace each table with a marker so we can recover its flattened text
    # while keeping document order via get_text().
    tables: list[str] = []
    for table in body.find_all("table"):
        flat = _flatten_table(table)
        marker = f"\nTABLE{len(tables)}\n"
        tables.append(flat)
        table.replace_with(marker)

    raw = body.get_text("\n")
    section = "Front matter"
    seen_item1 = False
    segments: list[tuple[str, str, bool]] = []

    for line in raw.split("\n"):       
        line = _clean(line)
        if not line:
            continue
        m = re.match(r"^TABLE(\d+)$", line)
        if m:
            segments.append((section, tables[int(m.group(1))], True))
            continue
        if _is_boilerplate(line):
            continue
        # Section header heuristic: short "Item N ..." line. Skip the table-of-
        # contents copy by only "locking on" once we've passed the first real
        # Item 1 body (TOC items appear first and in a tight cluster).
        hm = _ITEM_RE.match(line)
        if hm and len(line) < 90:
            num = hm.group(1).upper()
            title = hm.group(2).strip(" .")
            label = f"Item {num}" + (f". {title}" if title else "")
            if num == "1" and not title and not seen_item1:
                seen_item1 = True
            section = label
            continue
        segments.append((section, line, False))
    return segments


def chunk_filing(html: str, meta: dict) -> list[dict]:
    """Pack segments into ~CHUNK_WORDS chunks with overlap; tables stay atomic."""
    segs = _segments(html)
    chunks: list[dict] = []
    buf: list[str] = []
    buf_words = 0
    buf_section = segs[0][0] if segs else "Body"

    def emit(words: list[str], section: str) -> None:
        body = " ".join(words).strip()
        if len(body) < 40:  # drop tiny fragments
            return
        idx = len(chunks)
        # Contextual chunking: prepend company + section so number-dense table
        # chunks (which otherwise lack anchor words) still match queries like
        # "NVIDIA total revenue". This text is what we embed AND show as a source.
        text = f"{meta['company']} - {section}\n{body}"
        chunks.append({
            "id": f"{meta['ticker']}-{idx:04d}",
            "text": text,
            "metadata": {
                "company": meta["company"],
                "ticker": meta["ticker"],
                "filing_date": meta["filing_date"],
                "accession": meta["accession"],
                "section": section,
                "chunk_id": f"{meta['ticker']}-{idx:04d}",
            },
        })

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            emit(buf, buf_section)
            # carry overlap words into the next chunk for continuity
            buf = buf[-CHUNK_OVERLAP_WORDS:] if CHUNK_OVERLAP_WORDS else []
            buf_words = len(buf)

    for section, text, is_table in segs:
        if is_table:
            flush()
            buf, buf_words = [], 0  # tables don't inherit prose overlap
            rows = text.split("\n")
            cur: list[str] = []
            cur_words = 0
            for row in rows:
                w = row.split()
                if cur and cur_words + len(w) > CHUNK_WORDS:
                    emit(cur, section)
                    cur, cur_words = [], 0
                cur.extend(w)
                cur_words += len(w)
            if cur:
                emit(cur, section)
            buf_section = section
            continue

        if section != buf_section and buf_words > CHUNK_WORDS // 2:
            flush()
            buf, buf_words = [], 0
        buf_section = section
        words = text.split()
        buf.extend(words)
        buf_words += len(words)
        if buf_words >= CHUNK_WORDS:
            flush()
    flush()
    return chunks
