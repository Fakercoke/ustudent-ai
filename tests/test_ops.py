"""Operations dashboard security, privacy, diagnosis and accounting tests."""
from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone
import gzip
import json

from fastapi.testclient import TestClient

import app.rag.pipeline as pipeline
from app.agent.agent import OpsLLMCallback
from app.config import get_settings
from app.main import app
from app.ops.context import annotate_input, current_trace, record_llm_call, reset_trace, start_trace
from app.ops.store import save_eval_run, save_request, summary


def _enable(monkeypatch) -> tuple[TestClient, tuple[str, str]]:
    monkeypatch.setenv("OPS_ADMIN_USERNAME", "reviewer")
    monkeypatch.setenv("OPS_ADMIN_PASSWORD", "a-long-test-password")
    monkeypatch.setenv("OPS_HASH_SALT", "a-private-test-salt")
    monkeypatch.setenv("OPS_INSECURE_ALLOWED_CLIENTS", "testclient")
    get_settings.cache_clear()
    return TestClient(app), ("reviewer", "a-long-test-password")


def test_dashboard_is_disabled_without_a_deployment_password(client):
    response = client.get("/ops")
    assert response.status_code == 503


def test_dashboard_requires_valid_basic_auth(monkeypatch):
    client, auth = _enable(monkeypatch)
    assert client.get("/ops").status_code == 401
    response = client.get("/ops", auth=auth)
    assert response.status_code == 200
    assert "RAG 运营后台" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_public_http_ops_login_is_rejected(monkeypatch):
    _, auth = _enable(monkeypatch)
    public_client = TestClient(
        app,
        base_url="http://49.235.155.82",
        client=("203.0.113.7", 50000),
    )
    response = public_client.get("/ops", auth=auth, headers={"Host": "localhost"})
    assert response.status_code == 403
    assert "SSH tunnel" in response.json()["detail"]


def test_client_hash_trusts_only_configured_proxy(monkeypatch):
    monkeypatch.setenv("OPS_TRUSTED_PROXY_NETWORKS", "172.16.0.0/12")
    get_settings.cache_clear()
    proxy = TestClient(app, client=("172.19.0.5", 50000))
    proxy.post(
        "/echo",
        json={"message": "one"},
        headers={"X-Forwarded-For": "203.0.113.10"},
    )
    proxy.post(
        "/echo",
        json={"message": "two"},
        headers={"X-Forwarded-For": "203.0.113.11"},
    )
    assert summary(days=1)["overview"]["unique_callers"] == 2


def test_public_client_cannot_spoof_hash_with_forwarded_header(monkeypatch):
    monkeypatch.setenv("OPS_TRUSTED_PROXY_NETWORKS", "172.16.0.0/12")
    get_settings.cache_clear()
    public = TestClient(app, client=("198.51.100.9", 50000))
    for forwarded in ("203.0.113.10", "203.0.113.11"):
        public.post(
            "/echo",
            json={"message": forwarded},
            headers={"X-Forwarded-For": forwarded},
        )
    assert summary(days=1)["overview"]["unique_callers"] == 1


def test_far_rag_request_is_redacted_and_diagnosed(monkeypatch):
    client, auth = _enable(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda question, k=3: [{
            "text": "Unrelated office contact details.",
            "source": "data/handbook.md",
            "heading": "Contacts",
            "distance": 0.91,
        }],
    )

    response = client.post(
        "/rag-ask",
        json={"question": "Help jane@example.com with an unknown policy"},
    )
    assert response.status_code == 200

    data = client.get("/ops/api/summary?days=1", auth=auth).json()
    assert data["rag"]["requests"] == 1
    assert data["rag"]["fallbacks"] == 1
    assert data["recent"][0]["diagnosis"] == "distance_gate"
    assert "jane@example.com" not in data["recent"][0]["input_preview"]
    assert "[REDACTED_EMAIL]" in data["recent"][0]["input_preview"]


def test_token_cost_uses_configured_rates(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_COST_PER_MILLION", "2")
    monkeypatch.setenv("LLM_CACHED_INPUT_COST_PER_MILLION", "0.5")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_MILLION", "3")
    monkeypatch.setenv("LLM_COST_CURRENCY", "CNY")
    get_settings.cache_clear()

    token = start_trace("cost-1", "/rag-ask", "POST")
    annotate_input("How many credits?")
    record_llm_call(
        model="deepseek-test",
        prompt_tokens=100,
        cached_prompt_tokens=20,
        completion_tokens=10,
        total_tokens=110,
    )
    trace = current_trace()
    assert trace is not None
    save_request(trace, status_code=200, latency_ms=50, client_hash="anon")
    reset_trace(token)

    data = summary(days=1)
    assert data["llm"]["total_tokens"] == 110
    assert data["llm"]["estimated_cost"] == 0.0002
    assert data["llm"]["cost_configured"] is True


def test_latest_eval_is_separate_from_live_traffic():
    save_eval_run({
        "set": "golden",
        "n": 8,
        "refusal_accuracy": 1.0,
        "retrieval_hit": 1.0,
        "answer_grounded": 0.875,
    })
    data = summary(days=7)
    assert data["overview"]["requests"] == 0
    assert data["latest_evals"][0]["set"] == "golden"
    assert data["latest_evals"][0]["answer_grounded"] == 0.875


def test_markdown_report_is_downloadable(monkeypatch):
    client, auth = _enable(monkeypatch)
    response = client.get("/ops/report.md?days=7", auth=auth)
    assert response.status_code == 200
    assert "ustudent AI 运营报告" in response.text
    assert "attachment" in response.headers["content-disposition"]


def test_health_and_browser_assets_do_not_pollute_business_metrics(client):
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/favicon.ico").status_code == 404
    assert summary(days=1)["overview"]["requests"] == 0


def test_agent_callback_records_each_completion_and_error():
    token = start_trace("agent-usage", "/agent-chat", "POST")
    callback = OpsLLMCallback("agent-test-model", trace=current_trace())
    callback.on_llm_end(
        SimpleNamespace(
            llm_output={
                "model_name": "deepseek-test",
                "token_usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                },
            },
            generations=[],
        )
    )
    callback.on_llm_error(RuntimeError("provider unavailable"))
    trace = current_trace()
    assert trace is not None
    assert trace.llm_calls == 2
    assert trace.llm_errors == 1
    assert trace.total_tokens == 100
    assert trace.llm_models == ["deepseek-test", "agent-test-model"]
    reset_trace(token)


def test_nginx_page_views_are_separate_from_api_and_static_assets(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    access_log = tmp_path / "access.log"
    events = [
        {"time": now, "method": "GET", "uri": "/login", "status": 200},
        {"time": now, "method": "GET", "uri": "/dashboard", "status": 200},
        {"time": now, "method": "GET", "uri": "/assets/app.js", "status": 200},
        {"time": now, "method": "GET", "uri": "/api/courses", "status": 200},
        {"time": now, "method": "GET", "uri": "/wp-admin", "status": 200},
        {"time": now, "method": "GET", "uri": "/dashboard", "status": 500},
    ]
    access_log.write_text("\n".join(json.dumps(item) for item in events) + "\n")
    monkeypatch.setenv("OPS_WEB_ACCESS_LOG_PATH", str(access_log))
    get_settings.cache_clear()

    data = summary(days=1)
    assert data["web"]["available"] is True
    assert data["web"]["page_views"] == 2
    assert data["web"]["errors"] == 1
    assert {item["path"] for item in data["web"]["top_paths"]} == {
        "/login", "/dashboard"
    }


def test_nginx_page_views_include_rotated_and_compressed_logs(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    access_log = tmp_path / "access.log"
    access_log.write_text(
        json.dumps({"time": now, "method": "GET", "uri": "/login", "status": 200})
        + "\n"
    )
    with gzip.open(tmp_path / "access.log.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({
                "time": now,
                "method": "GET",
                "uri": "/ai-chat",
                "status": 200,
            })
            + "\n"
        )
    monkeypatch.setenv("OPS_WEB_ACCESS_LOG_PATH", str(access_log))
    get_settings.cache_clear()

    data = summary(days=1)
    assert data["web"]["page_views"] == 2
    assert data["web"]["partial"] is False


def test_nginx_log_limit_is_reported_as_partial(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    access_log = tmp_path / "access.log"
    access_log.write_text(
        ("not-json\n" * 2_000)
        + json.dumps({"time": now, "method": "GET", "uri": "/", "status": 200})
        + "\n"
    )
    monkeypatch.setenv("OPS_WEB_ACCESS_LOG_PATH", str(access_log))
    monkeypatch.setenv("OPS_WEB_LOG_MAX_BYTES", "10000")
    get_settings.cache_clear()

    data = summary(days=1)
    assert data["web"]["partial"] is True
    assert data["web"]["page_views"] == 1


def test_gzip_log_limit_counts_decompressed_bytes(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    access_log = tmp_path / "access.log"
    access_log.write_text("")
    event = json.dumps({
        "time": now,
        "method": "GET",
        "uri": "/dashboard",
        "status": 200,
    }) + "\n"
    with gzip.open(tmp_path / "access.log.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(event * 2_000)
    monkeypatch.setenv("OPS_WEB_ACCESS_LOG_PATH", str(access_log))
    monkeypatch.setenv("OPS_WEB_LOG_MAX_BYTES", "10000")
    get_settings.cache_clear()

    data = summary(days=1)
    assert data["web"]["partial"] is True
    assert 0 < data["web"]["page_views"] < 2_000
