"""Collect one request's diagnostics without coupling business code to SQLite.

FastAPI starts a trace in middleware.  The RAG pipeline and LLM client add
facts to the same in-memory object, then the middleware writes one row after
the response finishes.  Calls made directly from tests/scripts have no active
trace and therefore do not create misleading "web request" records.
"""
from __future__ import annotations

import json
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.safety import redact


@dataclass
class RequestTrace:
    request_id: str
    endpoint: str
    method: str
    input_preview: str = ""
    rag_seen: bool = False
    rag_top1_distance: float | None = None
    rag_distance_threshold: float = 0.75
    rag_source_headings: list[str] = field(default_factory=list)
    rag_used_fallback: bool = False
    rag_degraded: bool = False
    rag_blocked: bool = False
    tool_names: list[str] = field(default_factory=list)
    llm_models: list[str] = field(default_factory=list)
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_errors: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


_current: ContextVar[RequestTrace | None] = ContextVar(
    "ustudent_ops_request_trace", default=None
)


def start_trace(request_id: str, endpoint: str, method: str) -> Token:
    return _current.set(RequestTrace(request_id, endpoint, method))


def current_trace() -> RequestTrace | None:
    return _current.get()


def reset_trace(token: Token) -> None:
    _current.reset(token)


def annotate_input(text: str) -> None:
    """Store only a short, redacted preview—never raw request bodies."""
    trace = current_trace()
    if trace is None:
        return
    clean, _ = redact(re.sub(r"\s+", " ", text).strip())
    trace.input_preview = clean[:200]


def annotate_rag(result: Any, *, distance_threshold: float = 0.75) -> None:
    trace = current_trace()
    if trace is None:
        return
    sources = list(getattr(result, "sources", []) or [])
    trace.rag_seen = True
    trace.rag_distance_threshold = float(distance_threshold)
    trace.rag_top1_distance = (
        float(sources[0]["distance"]) if sources else None
    )
    trace.rag_source_headings = [
        str(source.get("heading", "Unknown")) for source in sources
    ]
    trace.rag_used_fallback = bool(getattr(result, "used_fallback", False))
    trace.rag_degraded = bool(getattr(result, "degraded", False))
    trace.rag_blocked = bool(getattr(result, "blocked", False))


def annotate_tools(tool_calls: list[dict[str, Any]]) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.tool_names.extend(
        str(call.get("name", "unknown")) for call in tool_calls
    )


def record_llm_call(
    *,
    model: str,
    prompt_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    cache_hit: bool = False,
    trace: RequestTrace | None = None,
) -> None:
    trace = trace or current_trace()
    if trace is None:
        return
    if model and model not in trace.llm_models:
        trace.llm_models.append(model)
    if cache_hit:
        trace.llm_cache_hits += 1
        return
    trace.llm_calls += 1
    trace.llm_errors += int(not success)
    trace.prompt_tokens += max(0, int(prompt_tokens or 0))
    trace.cached_prompt_tokens += max(0, int(cached_prompt_tokens or 0))
    trace.completion_tokens += max(0, int(completion_tokens or 0))
    reported_total = max(0, int(total_tokens or 0))
    trace.total_tokens += max(
        reported_total,
        max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0)),
    )


def serialise_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)
