"""Tests for the /rag-ask HTTP layer.

Pipeline behaviour (abstention, degradation, retry, input hygiene) lives in
tests/test_rag_pipeline_contract.py. This file only covers the endpoint:
request validation, response shape, and the gate that skips generation.
"""
import json

import pytest
from fastapi.testclient import TestClient

import app.rag.pipeline as pipeline
from app.main import app
from app.rag.pipeline import (
    DISTANCE_THRESHOLD,
    FALLBACK_ANSWER,
    build_context,
    rag_answer,
)

client = TestClient(app)


def hit(distance: float, heading: str = "Graduation") -> dict:
    return {
        "text": "You need a minimum of 120 credits.",
        "source": "data/handbook.md",
        "heading": heading,
        "distance": distance,
    }


@pytest.fixture
def stub(monkeypatch):
    def _stub(hits, can_answer=True, answer="You need 120 credits [1]."):
        monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: hits)
        monkeypatch.setattr(
            pipeline, "chat",
            lambda **kw: json.dumps({"can_answer": can_answer, "answer": answer}),
        )
    return _stub


# ---------------------------------------------------------- distance gate
def test_far_result_never_reaches_the_model(monkeypatch):
    """Burning tokens on a hopeless query is a bug, not just a cost."""
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: [hit(0.95)])

    def boom(**kw):
        raise AssertionError("the model must not be called past the threshold")

    monkeypatch.setattr(pipeline, "chat", boom)

    r = rag_answer("what AWS region")
    assert r.used_fallback is True
    assert r.answer == FALLBACK_ANSWER["default"]
    assert r.sources                      # still returned, for diagnosis


def test_empty_retrieval_falls_back(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: [])
    r = rag_answer("anything")
    assert r.used_fallback is True and r.sources == []


def test_threshold_boundary_is_inclusive(stub):
    """distance == threshold passes; only strictly greater is refused."""
    stub([hit(DISTANCE_THRESHOLD)])
    assert rag_answer("q").used_fallback is False


def test_close_result_is_answered(stub):
    stub([hit(0.28)])
    r = rag_answer("How many credits do I need to graduate?")
    assert r.used_fallback is False and "120 credits" in r.answer


# ------------------------------------------------------------ context format
def test_build_context_numbers_and_tags_each_chunk():
    ctx = build_context([hit(0.2, "Graduation"), hit(0.3, "Credit load")])
    assert "[1] (data/handbook.md § Graduation)" in ctx
    assert "[2] (data/handbook.md § Credit load)" in ctx


# ------------------------------------------------------------- HTTP surface
def test_endpoint_returns_the_documented_shape(stub):
    stub([hit(0.28)])
    body = client.post("/rag-ask", json={"question": "How many credits?"}).json()
    assert set(body) == {"answer", "sources", "used_fallback", "degraded", "blocked"}
    assert body["sources"][0]["heading"] == "Graduation"
    assert body["sources"][0]["distance"] == 0.28


def test_endpoint_reports_refusal(stub):
    stub([hit(0.28)], can_answer=False)
    body = client.post("/rag-ask", json={"question": "how do I appeal?"}).json()
    assert body["used_fallback"] is True and body["degraded"] is False


def test_endpoint_exposes_degraded_flag(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: [hit(0.28)])
    monkeypatch.setattr(pipeline, "chat",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    body = client.post("/rag-ask", json={"question": "How many credits?"}).json()
    assert body["degraded"] is True


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "hi", "k": 0},
    {"question": "hi", "k": 99},
])
def test_endpoint_rejects_bad_input(payload):
    assert client.post("/rag-ask", json=payload).status_code == 422


def test_endpoint_is_listed_in_openapi():
    assert "/rag-ask" in client.get("/openapi.json").json()["paths"]
