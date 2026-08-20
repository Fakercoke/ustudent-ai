"""FastAPI entrypoint for the ustudent-ai service.

Every new endpoint you write goes into a module under `app/routes/`
and is wired in below with `app.include_router(...)`.
"""
import hashlib
import ipaddress
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.ops.context import current_trace, reset_trace, start_trace
from app.ops.store import save_request
from app.routes import agent_chat, ask, can_graduate, echo, health, home, ops, rag_ask

log = logging.getLogger(__name__)

app = FastAPI(
    title="ustudent AI service",
    version="0.1.0",
)


def _client_identity(request: Request) -> str:
    """Use one proxy-supplied IP only when the direct peer is trusted.

    Public callers cannot spoof metrics with X-Forwarded-For.  The Tencent
    Nginx config replaces (rather than appends) that header with remote_addr.
    """
    direct = request.client.host if request.client else "unknown"
    try:
        direct_ip = ipaddress.ip_address(direct)
        trusted = any(
            direct_ip in ipaddress.ip_network(network.strip())
            for network in get_settings().ops_trusted_proxy_networks.split(",")
            if network.strip()
        )
    except ValueError:
        trusted = False
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if trusted and forwarded and "," not in forwarded:
        try:
            return str(ipaddress.ip_address(forwarded))
        except ValueError:
            pass
    return direct


@app.middleware("http")
async def operations_trace(request: Request, call_next):
    """Record one privacy-safe row per business request.

    Dashboard polling and health probes are excluded so they do not inflate
    usage.  The client address is salted and hashed; raw IPs and raw request
    bodies never enter the operations database.
    """
    # Do not trust a caller-supplied ID: duplicates could suppress telemetry
    # because request_id is unique in SQLite, and oversized values add noise.
    request_id = uuid.uuid4().hex
    token = start_trace(request_id, request.url.path, request.method)
    started = time.perf_counter()
    status_code = 500
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        trace = current_trace()
        non_business_paths = {
            "/health", "/favicon.ico", "/openapi.json", "/docs", "/redoc",
        }
        if trace is not None and not (
            request.url.path.startswith("/ops")
            or request.url.path in non_business_paths
        ):
            try:
                client = _client_identity(request)
                salt = get_settings().ops_hash_salt
                client_hash = hashlib.sha256(
                    f"{salt}|{client}".encode("utf-8")
                ).hexdigest()[:16]
                await run_in_threadpool(
                    save_request,
                    trace,
                    status_code=status_code,
                    latency_ms=elapsed_ms,
                    client_hash=client_hash,
                )
            except Exception:  # telemetry must never break the product
                log.exception("failed to persist operations trace")
        reset_trace(token)

app.include_router(home.router)
app.include_router(health.router, tags=["health"])
app.include_router(echo.router, tags=["echo"])
app.include_router(can_graduate.router, tags=["lesson-3"])
app.include_router(ask.router, tags=["ask"])
app.include_router(rag_ask.router, tags=["lesson-7 · 作品一 RAG"])
app.include_router(agent_chat.router, tags=["lesson-9 · 作品二 Agent"])
app.include_router(ops.router, tags=["operations"])
