"""Resolve and download each corpus company's 10-K from SEC EDGAR.

Accession numbers are pinned (see PINNED_ACCESSIONS) so a re-run reproduces the
exact filings the eval was built against; falls back to the most-recent 10-K if
a pin is cleared or unavailable.

Uses only the standard library so the download step has no third-party
dependencies. SEC asks unauthenticated scripts to send a descriptive
User-Agent with a contact email; we do.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

# SEC requires a contact User-Agent: "Sample Company Name AdminContact@example.com"
USER_AGENT = "Fairway-Takehome Ankush Rai ankushrai37@gmail.com"

# The fixed corpus. Tickers are used to resolve CIKs from SEC's ticker map.
COMPANIES = {
    "Apple Inc.": "AAPL",
    "JPMorgan Chase & Co.": "JPM",
    "Walmart Inc.": "WMT",
    "The Coca-Cola Company": "KO",
    "NVIDIA Corporation": "NVDA",
    "Caterpillar Inc.": "CAT",
}

# Pinned accession numbers for reproducibility: re-running this downloads the
# EXACT filings the eval was built against, not whatever is newest. If you want
# to refresh to the latest 10-Ks, clear this map (or set a ticker to None) and
# it falls back to "most recent". The brief wants graders on the same corpus.
PINNED_ACCESSIONS = {
    "AAPL": "0000320193-25-000079",
    "JPM": "0001628280-26-008131",
    "WMT": "0000104169-26-000055",
    "KO": "0001628280-26-010047",
    "NVDA": "0001045810-26-000021",
    "CAT": "0000018230-26-000008",
}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
PRIMARY_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class Filing:
    company: str
    ticker: str
    cik: int
    accession: str          # e.g. 0000320193-24-000123
    filing_date: str        # YYYY-MM-DD
    primary_doc: str        # filename of the primary 10-K document
    source_url: str
    local_path: str = ""


def _get(url: str, accept_json: bool = True) -> bytes:
    """GET with the required User-Agent header. Small backoff on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - probe code; surface and retry
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _ticker_to_cik() -> dict[str, int]:
    data = json.loads(_get(TICKER_MAP_URL))
    # company_tickers.json is {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def resolve_10k(company: str, ticker: str, cik: int) -> Filing:
    """Resolve the PINNED 10-K (for reproducibility), else the most recent.

    The submissions API returns the company's recent filings newest-first. We
    pick the pinned accession if one is set and present; otherwise the first
    10-K (= latest). This keeps a re-run on the same corpus the eval used.
    """
    subs = json.loads(_get(SUBMISSIONS_URL.format(cik=cik)))
    recent = subs["filings"]["recent"]
    forms = recent["form"]
    accessions = recent["accessionNumber"]

    idx = None
    pin = PINNED_ACCESSIONS.get(ticker)
    if pin and pin in accessions:
        idx = accessions.index(pin)
    elif pin:
        print(f"  WARNING: pinned accession {pin} not found in recent filings; "
              f"falling back to most-recent 10-K")
    if idx is None:
        idx = next((i for i, form in enumerate(forms) if form == "10-K"), None)
    if idx is None:
        raise ValueError(f"No 10-K found for {company} (CIK {cik})")

    accession = accessions[idx]
    primary_doc = recent["primaryDocument"][idx]
    filing_date = recent["filingDate"][idx]
    url = PRIMARY_DOC_URL.format(
        cik=cik, accession_nodash=accession.replace("-", ""), doc=primary_doc
    )
    return Filing(
        company=company,
        ticker=ticker,
        cik=cik,
        accession=accession,
        filing_date=filing_date,
        primary_doc=primary_doc,
        source_url=url,
    )


def download_filing(f: Filing) -> Filing:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    html = _get(f.source_url, accept_json=False)
    out = RAW_DIR / f"{f.ticker}_{f.accession}.html"
    out.write_bytes(html)
    f.local_path = str(out)
    print(f"  saved {f.company}: {len(html):,} bytes -> {out.name}")
    return f


def fetch_all(only: str | None = None) -> list[Filing]:
    """Resolve + download every corpus filing (or just one ticker if `only`)."""
    cik_map = _ticker_to_cik()
    filings: list[Filing] = []
    for company, ticker in COMPANIES.items():
        if only and ticker != only:
            continue
        cik = cik_map[ticker]
        print(f"{company} [{ticker}] CIK={cik}")
        f = resolve_10k(company, ticker, cik)
        pinned = " (pinned)" if PINNED_ACCESSIONS.get(ticker) == f.accession else ""
        print(f"  10-K{pinned}: accession={f.accession} filed={f.filing_date} doc={f.primary_doc}")
        download_filing(f)
        filings.append(f)
        time.sleep(0.3)  # be polite to SEC
    manifest = RAW_DIR.parent / "manifest.json"
    if filings:
        existing = []
        if manifest.exists() and only:
            existing = json.loads(manifest.read_text())
            existing = [m for m in existing if m["ticker"] != only]
        manifest.write_text(json.dumps(existing + [asdict(f) for f in filings], indent=2))
        print(f"manifest -> {manifest}")
    return filings


if __name__ == "__main__":
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else None
    fetch_all(only=only)
