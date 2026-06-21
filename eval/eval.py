"""Hand-checked evaluation harness.

Runs each Q/A pair in qa_pairs.yaml through the real pipeline and scores:
- should_refuse pairs   -> pass if the system refused.
- grounded pairs        -> pass if it did NOT refuse and every `must_include`
                           token appears in the answer (number-format tolerant).
Prints a per-question table, a summary, and the failures with reasons.

Run:  ANTHROPIC_API_KEY=... python eval/eval.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from generate import answer          # noqa: E402
from retrieve import Retriever       # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa_pairs.yaml"


def _norm(s: str) -> str:
    # lower-case, drop $ and thousands separators so "391,035" == "$391035"
    return re.sub(r"[,$]", "", s.lower())


def score(pair: dict, result: dict) -> tuple[bool, str]:
    refused = result["refused"]
    if pair.get("should_refuse"):
        return (refused, "refused as expected" if refused
                else "should have refused but answered")
    if refused:
        return (False, "refused but a grounded answer was expected")
    ans = _norm(result["answer"])
    missing = [t for t in pair.get("must_include", []) if _norm(str(t)) not in ans]
    if missing:
        return (False, f"missing required token(s): {missing}")
    return (True, "grounded answer contains expected facts")


def main() -> None:
    pairs = yaml.safe_load(QA_PATH.read_text())
    retriever = Retriever()
    rows, passes, failures = [], 0, []

    for p in pairs:
        res = answer(p["question"], retriever)
        ok, reason = score(p, res)
        passes += ok
        rows.append((p["id"], p["type"], ok, res["mode"], res["refused"]))
        if not ok:
            failures.append((p, res, reason))

    print("\n" + "=" * 66)
    print("  EVALUATION RESULTS - grounded QA over six 10-K filings")
    print("=" * 66)
    print("  RESULT  : PASS / FAIL against the expected answer")
    print("  MODE    : retrieval route (single-company | fan-out | open)")
    print("  REFUSED : did the system answer \"I don't know\"?")
    print("-" * 66)
    print(f"  {'ID':<5}{'TYPE':<18}{'RESULT':<8}{'MODE':<16}{'REFUSED'}")
    print("-" * 66)
    for pid, typ, ok, mode, refused in rows:
        print(f"  {pid:<5}{typ:<18}{'PASS' if ok else 'FAIL':<8}{mode:<16}{'yes' if refused else 'no'}")
    print("-" * 66)
    print(f"  SCORE: {passes}/{len(pairs)} correctly grounded")
    print("=" * 66)

    if failures:
        print("\nFAILURES (full detail):")
        for p, res, reason in failures:
            print(f"\n  [{p['id']}] {p['question']}")
            print(f"    why: {reason}")
            print(f"    got: {res['answer'][:240]}")
    print()


if __name__ == "__main__":
    main()
