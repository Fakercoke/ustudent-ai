"""Contract tests for the structured-output pipeline.

These cover the failure modes that string-matching prose could not:
a Chinese refusal, an empty payload, malformed JSON, and a model that claims
an answer without producing one.
"""
import json

import pytest

import app.rag.pipeline as pipeline
from app.rag.pipeline import (
    MAX_QUESTION_CHARS,
    RETRY_DISTANCE,
    clean_question,
    rag_answer,
)


def hit(distance=0.28, heading="Graduation"):
    return {
        "text": "You need a minimum of 120 credits.",
        "source": "data/handbook.md",
        "heading": heading,
        "distance": distance,
    }


@pytest.fixture
def wire(monkeypatch):
    """Stub retrieval and generation; `raw` is the model's literal payload."""
    def _wire(hits, raw):
        monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: hits)
        monkeypatch.setattr(pipeline, "chat", lambda **kw: raw)
    return _wire


def payload(can, answer=""):
    return json.dumps({"can_answer": can, "answer": answer})


# ------------------------------------------------- structured abstention
def test_refusal_is_read_from_a_field_not_from_prose(wire):
    wire([hit()], payload(False))
    assert rag_answer("how do I appeal a grade").used_fallback is True


def test_chinese_refusal_is_detected(wire):
    """The regression that motivated the change: the model refuses in Chinese,
    so `startswith("i don't know")` silently reported success."""
    wire([hit()], payload(False))
    r = rag_answer("我怎么申诉最终成绩")
    assert r.used_fallback is True
    assert "教务处" in r.answer          # fallback text matches the question's language


def test_english_question_gets_english_fallback(wire):
    wire([hit()], payload(False))
    assert "Academic Advising" in rag_answer("how do I appeal a grade").answer


def test_quoted_dont_know_no_longer_misfires(wire):
    """Previously this exact answer was flagged as a refusal."""
    wire([hit()], payload(True, "Log in. If you don't know your ID, contact IT [1]."))
    assert rag_answer("how do I log in?").used_fallback is False


# ------------------------------------------------------- bad model output
@pytest.mark.parametrize("raw", ["", "   ", "not json at all", "{}",
                                 '{"answer": "hi"}', "[1, 2, 3]"])
def test_unusable_payload_degrades_instead_of_returning_it(wire, raw):
    wire([hit()], raw)
    r = rag_answer("How many credits?")
    assert r.degraded is True
    assert "120 credits" in r.answer     # the retrieved material is handed over


def test_claiming_an_answer_without_one_degrades(wire):
    wire([hit()], payload(True, "   "))
    assert rag_answer("How many credits?").degraded is True


def test_generation_exception_degrades(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: [hit()])
    monkeypatch.setattr(pipeline, "chat",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("503")))
    r = rag_answer("How many credits?")
    assert r.degraded is True and r.used_fallback is True


# --------------------------------------------------- message roles
def test_rules_and_question_travel_in_separate_messages(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k=3: [hit()])
    monkeypatch.setattr(pipeline, "chat",
                        lambda **kw: seen.update(kw) or payload(True, "120 [1]"))
    rag_answer("How many credits?")
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user"]
    assert "MATERIAL" in seen["messages"][0]["content"]
    assert seen["messages"][1]["content"] == "How many credits?"
    assert seen["response_format"] == {"type": "json_object"}


# ------------------------------------------------------- input hygiene
def test_whitespace_is_collapsed():
    assert clean_question("  how   many\n\ncredits?  ") == "how many credits?"


def test_question_is_capped():
    assert len(clean_question("x" * 5000)) == MAX_QUESTION_CHARS


def test_blank_question_short_circuits(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not retrieve on an empty question")
    monkeypatch.setattr(pipeline, "retrieve", boom)
    assert rag_answer("   \n  ").used_fallback is True


def test_endpoint_rejects_overlong_question():
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).post("/rag-ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert r.status_code == 422


# --------------------------------------------- retry on weak retrieval
def test_weak_retrieval_triggers_a_forced_rewrite(monkeypatch):
    """`needs_rewrite` cannot see a typo inside a long well-formed sentence,
    so retrieval quality has to trigger the second attempt."""
    calls = []

    def fake_search(q, k=3, normalise=True):
        calls.append(q)
        return [hit(0.60)] if normalise else [hit(0.20)]

    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(pipeline, "needs_rewrite", lambda q: False)
    monkeypatch.setattr(pipeline, "rewrite_query", lambda q, force=False: "fixed query")

    hits = pipeline.retrieve("What are the gradution requirements for a degree here?")
    assert hits[0]["distance"] == 0.20
    assert calls[-1] == "fixed query"


def test_good_retrieval_does_not_pay_for_a_rewrite(monkeypatch):
    monkeypatch.setattr(pipeline, "search",
                        lambda q, k=3, normalise=True: [hit(RETRY_DISTANCE - 0.01)])
    monkeypatch.setattr(pipeline, "needs_rewrite", lambda q: False)

    def boom(*a, **kw):
        raise AssertionError("no rewrite when retrieval is already good")

    monkeypatch.setattr(pipeline, "rewrite_query", boom)
    assert pipeline.retrieve("a perfectly fine question?")[0]["distance"] < RETRY_DISTANCE
