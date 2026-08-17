"""Query normalisation — runs before retrieval.

The embedding model (all-MiniLM-L6-v2) is English-only and matches on wording,
not intent. Three real failure modes measured against the handbook index:

    "how can i gradute"      0.639  -> hits § Academic Advising Office (wrong)
    "graduate"               0.566  -> right section, but distance inflated
    "毕业需要多少学分"          0.684  -> hits § FAQ index page (wrong)

Rewriting the query into one well-formed English sentence fixes all three
without swapping the embedding model (which would pull in torch and grow the
image from ~320 MB to >1.5 GB).

Two safety properties, both deliberate:

  * The rewrite is used **only for retrieval**. The answer prompt always
    receives the user's original wording. This reduces direct question-drift,
    although a bad rewrite can still retrieve misleading evidence.
  * If the LLM is unavailable, `rewrite_query` returns the original text
    rather than raising — retrieval degrades, it does not break.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

_PROMPT = """Rewrite the student's question into ONE clear, well-formed English search
query for a university student handbook.

Rules:
- Fix spelling mistakes.
- Translate to English if the question is in another language.
- Keep the original meaning. Do NOT add topics the student did not mention.
- If the question is already a clear English sentence, return it unchanged.
- Output ONLY the rewritten query. No quotes, no explanation.

Student's question: {q}"""

# CJK, Hangul, Hiragana/Katakana, Cyrillic, Arabic, Thai, Devanagari
_NON_LATIN = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯"
    r"Ѐ-ӿ؀-ۿ฀-๿ऀ-ॿ]"
)


def detect_language(text: str) -> str:
    """Coarse label used to tell the answering model which language to reply in."""
    if re.search(r"[一-鿿㐀-䶿]", text):
        return "Chinese"
    if re.search(r"[가-힯]", text):
        return "Korean"
    if re.search(r"[぀-ヿ]", text):
        return "Japanese"
    if _NON_LATIN.search(text):
        return "the user's language"
    return "English"


def needs_rewrite(question: str) -> bool:
    """Skip the LLM call for questions that are already well-formed English.

    Saves a round trip on the common case. A question is left alone when it is
    latin-script, has enough words to carry intent, and ends like a question.
    """
    q = question.strip()
    if _NON_LATIN.search(q):
        return True                       # 非拉丁字母 -> 需要翻译
    if len(q.split()) < 5:
        return True                       # 太短 -> 需要补全
    if not q.endswith("?"):
        return True                       # 不像完整问句 -> 可能是口语/关键词
    return False


@lru_cache(maxsize=512)
def rewrite_query(question: str, force: bool = False) -> str:
    """Normalise `question` for retrieval. Never raises; falls back to the input.

    `force=True` skips the `needs_rewrite` heuristic. Callers use it when
    retrieval came back weak despite the question looking well formed — the
    heuristic cannot see a typo buried in a long, otherwise correct sentence.
    """
    if not force and not needs_rewrite(question):
        return question

    try:
        from app.config import get_settings
        from app.llm import chat  # imported lazily so this module works without a key

        # Rewriting is mechanical (spelling, translation) — route it to the
        # cheap model when one is configured. See Settings.llm_small_model.
        out = chat(
            messages=[{"role": "user", "content": _PROMPT.format(q=question)}],
            temperature=0.0,
            model=get_settings().llm_small_model or None,
        ).strip().strip('"').strip()
    except Exception as exc:                       # noqa: BLE001 — degrade, don't break
        log.warning("query rewrite failed (%s); falling back to original", exc)
        return question

    # A rewrite that came back empty, or ballooned into a paragraph, is not
    # trustworthy — fall back rather than search for something odd.
    if not out or len(out) > 300 or "\n" in out:
        log.warning("query rewrite looked wrong (%r); falling back", out[:80])
        return question
    return out
