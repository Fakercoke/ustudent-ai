"""Lesson 10 · LLM-as-judge over the golden set.

    python lessons/lesson-10-eval-safety/starter/eval.py
    python lessons/lesson-10-eval-safety/starter/eval.py dev

Why a judge instead of string matching: `scripts/eval_rag.py` asks whether a
key phrase survived into the answer. That measures whether the right material
was retrieved, and nothing about whether the answer is correct — a system that
parrots "120 credits" inside a wrong sentence still scores a pass. A judge
reads the reference and the answer and decides.

Two properties make the score trustworthy enough to act on:

  temperature=0   the same answer must always receive the same verdict,
                  otherwise a re-run measures the judge's mood, not the system.
  disk cache      app/llm.py caches deterministic calls, so re-running after a
                  code change is free and instant for unchanged answers.

The judge is still an LLM and can be wrong. `--show` prints its reasoning so a
disputed verdict can be checked by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.llm import ask_llm            # noqa: E402
from app.rag.parse import parse_json_safe  # noqa: E402
from app.rag.pipeline import rag_answer    # noqa: E402

SETS = {
    "golden": ROOT / "data/golden/rag-eval.json",
    "dev": ROOT / "data/golden/dev-set.json",
}

JUDGE_PROMPT = """You grade a university handbook assistant. Decide whether the \
SYSTEM ANSWER is acceptable given the REFERENCE answer.

Rules:
1. Correct if the system answer states the same facts as the reference. Wording,
   length and language may differ — a Chinese answer to an English reference is
   fine when the facts match.
2. If REFERENCE is "<no answer in handbook>", then the system answer is CORRECT
   only when it declines — says it does not know, that the handbook does not
   cover this, or refers the student elsewhere. Any attempt to answer the
   question is INCORRECT.
3. If REFERENCE contains a fact and the system answer omits or contradicts it,
   that is INCORRECT.
4. Extra correct detail beyond the reference is fine.

Return ONLY a JSON object, no prose and no code fence:
{{"correct": true, "reason": "<one short sentence>"}}

QUESTION: {question}

REFERENCE: {reference}

SYSTEM ANSWER: {answer}"""


def judge(question: str, reference: str, answer: str) -> dict:
    """Grade one answer. Any unreadable judge output counts as a failure.

    Defaulting to False is the conservative choice: a broken judge should make
    the score look worse than reality, never better. A silent pass would be the
    one failure mode that hides real regressions.
    """
    raw = ask_llm(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
        temperature=0.0,
    )
    data = parse_json_safe(raw)
    if not data or "correct" not in data:
        return {"correct": False, "reason": f"judge returned unparseable output: {raw[:80]!r}"}
    return {"correct": bool(data["correct"]), "reason": str(data.get("reason", ""))}


def reference_for(item: dict) -> str:
    """The course set ships `ref`; the dev set describes the expected fact."""
    if item.get("ref"):
        return item["ref"]
    if not item["in_handbook"]:
        return "<no answer in handbook>"
    return item.get("must_contain") or "<see handbook>"


def run(which: str = "golden", show_reasons: bool = False) -> float:
    items = json.loads(SETS[which].read_text())["items"]
    correct = 0
    rows = []

    print(f"\n{'=' * 92}\nLLM-as-judge · {which} ({len(items)} 题) · temperature=0\n{'=' * 92}")
    for item in items:
        # temperature=0 on the system too, not just the judge. A deterministic
        # judge grading a non-deterministic answer still gives a score that
        # moves between runs — and the cache makes re-runs cost nothing.
        result = rag_answer(item["q"], temperature=0.0)
        reference = reference_for(item)
        verdict = judge(item["q"], reference, result.answer)
        correct += verdict["correct"]
        rows.append((item, result, verdict))

        flag = "OK" if verdict["correct"] else "XX"
        state = ("blocked" if result.blocked else
                 "degraded" if result.degraded else
                 "refused" if result.used_fallback else "answered")
        print(f"  {flag}  {item['id']:4} {state:9} {item['q'][:46]:48}")
        if show_reasons or not verdict["correct"]:
            print(f"        judge: {verdict['reason'][:110]}")

    score = correct / len(items)
    print(f"{'-' * 92}\n=== {correct}/{len(items)}  ({score:.0%}) ===")
    return score


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    which = args[0] if args else "golden"
    if which not in SETS:
        print(f"Usage: python eval.py [{'|'.join(SETS)}] [--show]")
        raise SystemExit(1)
    run(which, show_reasons="--show" in sys.argv)
