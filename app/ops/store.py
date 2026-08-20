"""SQLite persistence and read models for the operations dashboard."""
from __future__ import annotations

import gzip
import json
import math
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from stat import S_ISREG
from typing import Any

from app.config import get_settings
from app.ops.context import RequestTrace, serialise_list


def _db_path() -> Path:
    return Path(get_settings().ops_db_path)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS request_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            client_hash TEXT NOT NULL,
            input_preview TEXT NOT NULL DEFAULT '',
            rag_seen INTEGER NOT NULL DEFAULT 0,
            rag_top1_distance REAL,
            rag_source_headings TEXT NOT NULL DEFAULT '[]',
            rag_used_fallback INTEGER NOT NULL DEFAULT 0,
            rag_degraded INTEGER NOT NULL DEFAULT 0,
            rag_blocked INTEGER NOT NULL DEFAULT 0,
            diagnosis TEXT NOT NULL,
            tool_names TEXT NOT NULL DEFAULT '[]',
            llm_models TEXT NOT NULL DEFAULT '[]',
            llm_calls INTEGER NOT NULL DEFAULT 0,
            llm_cache_hits INTEGER NOT NULL DEFAULT 0,
            llm_errors INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_request_events_created_at
            ON request_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_request_events_endpoint
            ON request_events(endpoint);
        CREATE INDEX IF NOT EXISTS idx_request_events_diagnosis
            ON request_events(diagnosis);

        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            set_name TEXT NOT NULL,
            question_count INTEGER NOT NULL,
            refusal_accuracy REAL NOT NULL,
            retrieval_hit REAL NOT NULL,
            answer_grounded REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at
            ON eval_runs(created_at);
        """
    )
    connection.commit()


def diagnose(trace: RequestTrace, status_code: int) -> str:
    """Classify the failed layer from facts we can actually observe.

    It deliberately never labels a successful-looking answer "correct".  Live
    traffic has no reference answer; correctness belongs to the eval set.
    """
    if status_code >= 500:
        return "system_error"
    if status_code >= 400:
        return "invalid_request"
    if trace.rag_blocked:
        return "security_block"
    if trace.rag_degraded:
        return "generation_failure"
    if trace.llm_errors:
        return "llm_error"
    if trace.rag_seen:
        if trace.rag_top1_distance is None:
            return "retrieval_empty"
        if (
            trace.rag_used_fallback
            and trace.rag_top1_distance > trace.rag_distance_threshold
        ):
            return "distance_gate"
        if trace.rag_used_fallback:
            return "model_abstention"
        return "rag_answered"
    if trace.tool_names:
        return "agent_tool_used"
    return "ok"


def _estimate_cost(trace: RequestTrace) -> float:
    settings = get_settings()
    cached = min(trace.cached_prompt_tokens, trace.prompt_tokens)
    uncached = max(0, trace.prompt_tokens - cached)
    return (
        uncached * settings.llm_input_cost_per_million
        + cached * settings.llm_cached_input_cost_per_million
        + trace.completion_tokens * settings.llm_output_cost_per_million
    ) / 1_000_000


def save_request(
    trace: RequestTrace,
    *,
    status_code: int,
    latency_ms: float,
    client_hash: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    values = (
        trace.request_id,
        now,
        trace.endpoint,
        trace.method,
        status_code,
        round(latency_ms, 3),
        client_hash,
        trace.input_preview,
        int(trace.rag_seen),
        trace.rag_top1_distance,
        serialise_list(trace.rag_source_headings),
        int(trace.rag_used_fallback),
        int(trace.rag_degraded),
        int(trace.rag_blocked),
        diagnose(trace, status_code),
        serialise_list(trace.tool_names),
        serialise_list(trace.llm_models),
        trace.llm_calls,
        trace.llm_cache_hits,
        trace.llm_errors,
        trace.prompt_tokens,
        trace.cached_prompt_tokens,
        trace.completion_tokens,
        trace.total_tokens,
        _estimate_cost(trace),
    )
    with _connect() as connection:
        retention_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=max(1, get_settings().ops_retention_days))
        ).isoformat()
        connection.execute(
            "DELETE FROM request_events WHERE created_at < ?",
            (retention_cutoff,),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO request_events (
                request_id, created_at, endpoint, method, status_code,
                latency_ms, client_hash, input_preview, rag_seen,
                rag_top1_distance, rag_source_headings, rag_used_fallback,
                rag_degraded, rag_blocked, diagnosis, tool_names, llm_models,
                llm_calls, llm_cache_hits, llm_errors, prompt_tokens,
                cached_prompt_tokens, completion_tokens, total_tokens,
                estimated_cost
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )


def save_eval_run(metrics: dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO eval_runs (
                created_at, set_name, question_count, refusal_accuracy,
                retrieval_hit, answer_grounded
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                str(metrics["set"]),
                int(metrics["n"]),
                float(metrics["refusal_accuracy"]),
                float(metrics["retrieval_hit"]),
                float(metrics["answer_grounded"]),
            ),
        )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 4)


def _json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
        return [str(item) for item in value] if isinstance(value, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


_WEB_TRAFFIC_CACHE: dict[
    tuple[Any, ...],
    tuple[float, tuple[tuple[str, int, int], ...], dict[str, Any]],
] = {}


def _web_log_files(path: Path) -> tuple[list[tuple[Path, int, int]], bool]:
    """Snapshot active/rotated logs; tolerate logrotate changing them mid-read."""
    snapshots: list[tuple[Path, int, int]] = []
    changed_during_scan = False
    try:
        candidates = list(path.parent.glob(f"{path.name}*"))
    except OSError:
        return [], True
    for candidate in candidates:
        if not (
            candidate.name == path.name
            or candidate.name.startswith(f"{path.name}.")
        ):
            continue
        try:
            file_stat = candidate.stat()
        except OSError:
            changed_during_scan = True
            continue
        if S_ISREG(file_stat.st_mode):
            snapshots.append((candidate, file_stat.st_size, file_stat.st_mtime_ns))
    snapshots.sort(key=lambda item: item[2], reverse=True)
    return snapshots, changed_during_scan


def _web_traffic(days: int) -> dict[str, Any]:
    """Read privacy-minimal JSON access logs without treating scanners as users."""
    settings = get_settings()
    unavailable = {
        "available": False,
        "partial": False,
        "page_views": 0,
        "top_paths": [],
        "errors": 0,
    }
    if not settings.ops_web_access_log_path:
        return unavailable
    path = Path(settings.ops_web_access_log_path)
    files, changed_during_scan = _web_log_files(path)
    if not files:
        unavailable["partial"] = changed_during_scan
        return unavailable

    allowed_routes = {
        route.strip() or "/"
        for route in settings.ops_web_routes.split(",")
        if route.strip()
    }
    signatures = tuple(
        (str(candidate), modified_ns, size)
        for candidate, size, modified_ns in files
    )
    cache_key = (
        days,
        tuple(sorted(allowed_routes)),
        settings.ops_web_log_max_bytes,
        str(path),
    )
    cached = _WEB_TRAFFIC_CACHE.get(cache_key)
    if cached is not None:
        cached_at, cached_signatures, cached_result = cached
        if (
            signatures == cached_signatures
            or time.monotonic() - cached_at
            < max(1, settings.ops_web_cache_seconds)
        ):
            return dict(cached_result)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    path_counts: Counter[str] = Counter()
    errors = 0
    max_bytes = max(10_000, settings.ops_web_log_max_bytes)
    remaining_bytes = max_bytes
    partial = changed_during_scan

    for file_index, (candidate, snapshot_size, _) in enumerate(files):
        if remaining_bytes <= 0:
            partial = True
            break

        is_gzip = candidate.suffix == ".gz"
        start_offset = 0
        if not is_gzip and snapshot_size > remaining_bytes:
            partial = True
            start_offset = snapshot_size - remaining_bytes

        try:
            handle = gzip.open(candidate, "rb") if is_gzip else candidate.open("rb")
            with handle:
                # A normal active log can be tailed efficiently.  Gzip files
                # must be streamed, and their *decompressed* bytes consume the
                # same global budget, preventing tiny zip-bomb-style inputs
                # from bypassing OPS_WEB_LOG_MAX_BYTES.
                file_bytes_left = (
                    remaining_bytes
                    if is_gzip
                    else min(remaining_bytes, snapshot_size - start_offset)
                )
                if start_offset:
                    handle.seek(start_offset)
                    discarded = handle.readline(file_bytes_left + 1)
                    used = min(len(discarded), file_bytes_left)
                    remaining_bytes -= used
                    file_bytes_left -= used

                while remaining_bytes > 0 and file_bytes_left > 0:
                    raw = handle.readline(min(remaining_bytes, file_bytes_left) + 1)
                    if not raw:
                        break
                    if len(raw) > remaining_bytes or len(raw) > file_bytes_left:
                        partial = True
                        remaining_bytes = 0
                        break
                    remaining_bytes -= len(raw)
                    file_bytes_left -= len(raw)
                    try:
                        event = json.loads(raw.decode("utf-8"))
                        timestamp = datetime.fromisoformat(str(event["time"]))
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        uri = str(event.get("uri", "/")).split("?", 1)[0]
                        method = str(event.get("method", "GET"))
                        status = int(event.get("status", 0))
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        KeyError,
                        ValueError,
                        TypeError,
                    ):
                        continue
                    if timestamp < since or method != "GET" or uri not in allowed_routes:
                        continue
                    if 200 <= status < 400:
                        path_counts[uri] += 1
                    elif status >= 500:
                        errors += 1
                if remaining_bytes <= 0 and handle.read(1):
                    partial = True
        except (OSError, EOFError, gzip.BadGzipFile):
            # Rotation may rename/compress a file between snapshot and open.
            # A broken historical gzip must not make the admin page return 500.
            partial = True
            continue

        if remaining_bytes <= 0 and file_index < len(files) - 1:
            partial = True
            break

    result = {
        "available": True,
        "partial": partial,
        "page_views": sum(path_counts.values()),
        "top_paths": [
            {"path": route, "count": count}
            for route, count in path_counts.most_common(8)
        ],
        "errors": errors,
    }
    # Log signatures invalidate this tiny cache automatically.  Keeping only
    # the newest entry prevents an ever-growing in-process cache.
    _WEB_TRAFFIC_CACHE.clear()
    _WEB_TRAFFIC_CACHE[cache_key] = (time.monotonic(), signatures, result)
    return dict(result)


def summary(days: int = 7, recent_limit: int = 50) -> dict[str, Any]:
    since_datetime = datetime.now(timezone.utc) - timedelta(days=days)
    since = since_datetime.isoformat()
    with _connect() as connection:
        retention_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=max(1, get_settings().ops_retention_days))
        ).isoformat()
        connection.execute(
            "DELETE FROM request_events WHERE created_at < ?",
            (retention_cutoff,),
        )
        connection.commit()
        rows = connection.execute(
            "SELECT * FROM request_events WHERE created_at >= ? ORDER BY id DESC",
            (since,),
        ).fetchall()
        eval_rows = connection.execute(
            """
            SELECT e.* FROM eval_runs e
            JOIN (
                SELECT set_name, MAX(id) AS max_id FROM eval_runs GROUP BY set_name
            ) latest ON latest.max_id = e.id
            ORDER BY e.set_name
            """
        ).fetchall()

    total = len(rows)
    rag_rows = [row for row in rows if row["rag_seen"]]
    distances = [
        float(row["rag_top1_distance"])
        for row in rag_rows if row["rag_top1_distance"] is not None
    ]
    headings: Counter[str] = Counter()
    diagnoses: Counter[str] = Counter()
    for row in rows:
        diagnoses[row["diagnosis"]] += 1
        headings.update(_json_list(row["rag_source_headings"]))

    daily: dict[str, dict[str, Any]] = {}
    for row in reversed(rows):
        date = row["created_at"][:10]
        item = daily.setdefault(
            date,
            {"date": date, "requests": 0, "rag_requests": 0,
             "fallbacks": 0, "tokens": 0, "latency_total": 0.0},
        )
        item["requests"] += 1
        item["rag_requests"] += int(row["rag_seen"])
        item["fallbacks"] += int(row["rag_used_fallback"])
        item["tokens"] += int(row["total_tokens"])
        item["latency_total"] += float(row["latency_ms"])
    for item in daily.values():
        item["avg_latency_ms"] = round(
            item.pop("latency_total") / item["requests"], 1
        )

    settings = get_settings()
    currency = settings.llm_cost_currency.upper()
    return {
        "window_days": days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "requests": total,
            "unique_callers": len({row["client_hash"] for row in rows}),
            "successful_requests": sum(row["status_code"] < 400 for row in rows),
            "server_errors": sum(row["status_code"] >= 500 for row in rows),
            "avg_latency_ms": round(
                sum(float(row["latency_ms"]) for row in rows) / total, 1
            ) if total else 0.0,
        },
        "web": _web_traffic(days),
        "rag": {
            "requests": len(rag_rows),
            "answered": sum(
                not row["rag_used_fallback"] and not row["rag_blocked"]
                for row in rag_rows
            ),
            "fallbacks": sum(row["rag_used_fallback"] for row in rag_rows),
            "fallback_rate": round(
                sum(row["rag_used_fallback"] for row in rag_rows) / len(rag_rows), 4
            ) if rag_rows else 0.0,
            "degraded": sum(row["rag_degraded"] for row in rag_rows),
            "blocked": sum(row["rag_blocked"] for row in rag_rows),
            "avg_top1_distance": round(sum(distances) / len(distances), 4)
                if distances else None,
            "p95_top1_distance": _percentile(distances, 0.95),
        },
        "llm": {
            "calls": sum(row["llm_calls"] for row in rows),
            "cache_hits": sum(row["llm_cache_hits"] for row in rows),
            "errors": sum(row["llm_errors"] for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "estimated_cost": round(sum(row["estimated_cost"] for row in rows), 6),
            "currency": currency,
            "cost_note": settings.llm_cost_note,
            "cost_configured": any((
                settings.llm_input_cost_per_million,
                settings.llm_cached_input_cost_per_million,
                settings.llm_output_cost_per_million,
            )),
        },
        "diagnoses": [
            {"name": name, "count": count}
            for name, count in diagnoses.most_common()
        ],
        "top_sources": [
            {"heading": heading, "count": count}
            for heading, count in headings.most_common(8)
        ],
        "daily": list(daily.values()),
        "latest_evals": [
            {
                "set": row["set_name"],
                "ran_at": row["created_at"],
                "questions": row["question_count"],
                "refusal_accuracy": row["refusal_accuracy"],
                "retrieval_hit": row["retrieval_hit"],
                "answer_grounded": row["answer_grounded"],
            }
            for row in eval_rows
        ],
        "recent": [
            {
                "time": row["created_at"],
                "request_id": row["request_id"],
                "endpoint": row["endpoint"],
                "status": row["status_code"],
                "latency_ms": row["latency_ms"],
                "input_preview": row["input_preview"],
                "top1_distance": row["rag_top1_distance"],
                "fallback": bool(row["rag_used_fallback"]),
                "diagnosis": row["diagnosis"],
                "tools": _json_list(row["tool_names"]),
                "tokens": row["total_tokens"],
            }
            for row in rows[:recent_limit]
        ],
    }
