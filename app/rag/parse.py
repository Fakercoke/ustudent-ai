"""Tolerant JSON extraction from model output.

Even in JSON mode a model occasionally wraps its object in a ```json fence, or
prefixes it with a sentence. Callers that need a dict should not have to guess
which shape they got, and must never mistake a parse failure for a valid
answer — so this returns None on anything it cannot read, and the caller
decides what a failure means.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def parse_json_safe(raw: str | None) -> dict | None:
    """Return the JSON object in `raw`, or None if there isn't a usable one.

    Handles, in order: a clean object, a ```json fenced block, and an object
    embedded in surrounding prose. Anything that is not a dict — a bare list,
    a number, a string — counts as unusable, because every caller here expects
    named fields.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _candidates(text: str):
    """Yield progressively more forgiving slices of `text`."""
    yield text

    fenced = _FENCE.search(text)
    if fenced:
        yield fenced.group(1)

    # Outermost braces — catches "Here is the result: {...} hope that helps".
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]
