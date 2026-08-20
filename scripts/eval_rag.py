#!/usr/bin/env python3
"""End-to-end evaluation of /rag-ask against a question set.

    python scripts/eval_rag.py dev       # tune against this
    python scripts/eval_rag.py golden    # held out — run once, at the end
    python scripts/eval_rag.py both

Two things are measured, and they fail for different reasons:

  refusal accuracy  did `used_fallback` match `in_handbook`?
                    Catches both halves of the bug: answering what the handbook
                    does not cover, and refusing what it does.

  answer grounding  for answerable questions, does the answer contain the fact
                    the corpus states? String matching, so it measures whether
                    the right material reached the model and survived into the
                    answer. It cannot judge fluency or completeness.

Retrieval hit rate is reported alongside so a failure can be attributed:
material missing from `sources` is a retrieval problem, material present but
absent from the answer is a generation problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import rag_answer  # noqa: E402
from app.ops.store import save_eval_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "dev": ROOT / "data/golden/dev-set.json",
    "golden": ROOT / "data/golden/rag-eval.json",
}


def key_fact(item: dict) -> str | None:
    """dev-set states it directly; the course's golden set needs a mapping."""
    if "must_contain" in item:
        return item["must_contain"]
    return {
        "g1": "120 credits", "g2": "18 credits",
        "g3": "30 completed credit hours", "g4": "credit-weighted average",
        "g5": "do not affect GPA", "g6": "full refund and no academic record",
    }.get(item["id"])


#: Short, language-neutral tokens for the course's golden set, which only ships
#: `ref` prose. Kept beside the retrieval phrases in `key_fact` so the two
#: checks stay visibly separate.
GOLDEN_ANSWER_TOKENS = {
    "g1": ["120"], "g2": ["18"], "g3": ["sophomore", "30"],
    "g4": ["credit-weighted", "weighted", "grade point"],
    "g5": ["W", "transcript"], "g6": ["Week 2"],
}


def answer_tokens(item: dict, fact: str) -> list[str]:
    """Acceptable substrings for the *answer*, as opposed to the sources."""
    return item.get("answer_any_of") or GOLDEN_ANSWER_TOKENS.get(item["id"]) or [fact]


def retrieval_contains_expected_fact(item: dict, fact: str, sources: list[dict]) -> bool:
    """Require the fact and, when specified, the expected heading in one chunk.

    Identifier-like facts such as ``CS101`` often occur in prerequisite and
    notes chunks.  Counting any mention as a hit gave a false positive when
    the actual CS101 course section ranked fourth and never reached Top-3.
    """
    expected_headings = [
        heading.casefold()
        for heading in item.get("expected_heading_any_of", [])
    ]
    for source in sources:
        if fact not in source["text"]:
            continue
        if not expected_headings or any(
            heading in source.get("heading", "").casefold()
            for heading in expected_headings
        ):
            return True
    return False


def run(name: str, verbose: bool = True) -> dict:
    items = json.loads(SETS[name].read_text())["items"]
    rows, refusal_ok, grounded, retrieved, answerable = [], 0, 0, 0, 0

    for it in items:
        r = rag_answer(it["q"])
        should_refuse = not it["in_handbook"]
        ref_ok = r.used_fallback == should_refuse
        refusal_ok += ref_ok

        fact, in_answer, in_sources = key_fact(it), None, None
        if it["in_handbook"] and fact:
            answerable += 1
            # Retrieval is checked against the verbatim corpus phrase.
            in_sources = retrieval_contains_expected_fact(it, fact, r.sources)
            # The answer is checked against short, language-neutral tokens.
            # An English phrase can never appear inside a Chinese answer, so
            # reusing `fact` here would score every 中文 question as a failure.
            in_answer = any(t in r.answer for t in answer_tokens(it, fact))
            retrieved += in_sources
            grounded += in_answer

        rows.append((it, r, ref_ok, in_sources, in_answer))

    if verbose:
        print(f"\n{'=' * 96}\n评测集: {name}  ({len(items)} 题)\n{'=' * 96}")
        print(f"{'id':5}{'拒答':>6}{'检索':>6}{'落答案':>8}  {'距离':>7}  问题")
        print("-" * 96)
        for it, r, ref_ok, in_src, in_ans in rows:
            d = r.sources[0]["distance"] if r.sources else 0
            mark = lambda v: "  -  " if v is None else ("  ✅ " if v else "  ❌ ")  # noqa: E731
            print(f"{it['id']:5}{'  ✅ ' if ref_ok else '  ❌ ':>6}"
                  f"{mark(in_src):>6}{mark(in_ans):>8}{d:>8.3f}  {it['q'][:52]}")
        print("-" * 96)

    m = {
        "set": name,
        "n": len(items),
        "refusal_accuracy": refusal_ok / len(items),
        "retrieval_hit": retrieved / answerable if answerable else 0.0,
        "answer_grounded": grounded / answerable if answerable else 0.0,
    }
    if verbose:
        print(f"  拒答准确率  {m['refusal_accuracy']:.0%}   ({refusal_ok}/{len(items)})")
        print(f"  检索命中率  {m['retrieval_hit']:.0%}   ({retrieved}/{answerable})")
        print(f"  答案落地率  {m['answer_grounded']:.0%}   ({grounded}/{answerable})")
    return m


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    names = ["dev", "golden"] if which == "both" else [which]
    if any(n not in SETS for n in names):
        print("Usage: python scripts/eval_rag.py [dev|golden|both]")
        raise SystemExit(1)

    results = [run(n) for n in names]
    for metrics in results:
        save_eval_run(metrics)
    print("\n  评测结果已写入 /ops 运营后台。")

    if len(results) > 1:
        print(f"\n{'=' * 96}\n对比\n{'=' * 96}")
        print(f"{'评测集':10}{'题数':>6}{'拒答准确':>10}{'检索命中':>10}{'答案落地':>10}")
        for m in results:
            print(f"{m['set']:10}{m['n']:>6}{m['refusal_accuracy']:>10.0%}"
                  f"{m['retrieval_hit']:>10.0%}{m['answer_grounded']:>10.0%}")
        print("\n  dev 用于调参，golden 是留出集。两者的差值即为过拟合幅度。")
