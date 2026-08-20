"""Tests for app/rag/query.py — query normalisation before retrieval."""
import app.rag.query as query_mod
from app.rag.query import detect_language, needs_rewrite, rewrite_query


# ------------------------------------------------------------ detect_language
def test_detect_language_chinese():
    assert detect_language("毕业需要多少学分") == "Chinese"


def test_detect_language_english():
    assert detect_language("How many credits do I need?") == "English"


def test_detect_language_mixed_counts_as_chinese():
    assert detect_language("CS201 可以选吗") == "Chinese"


# -------------------------------------------------------------- needs_rewrite
def test_well_formed_english_question_is_left_alone():
    """The common case must not pay for an LLM round trip."""
    assert needs_rewrite("How many credits do I need to graduate?") is False


def test_short_query_needs_rewrite():
    assert needs_rewrite("graduate") is True


def test_non_english_needs_rewrite():
    assert needs_rewrite("毕业需要多少学分") is True


def test_statement_without_question_mark_needs_rewrite():
    assert needs_rewrite("tell me about the refund policy please") is True


# --------------------------------------------------------------- rewrite_query
def test_rewrite_skipped_for_good_question_does_not_call_llm(monkeypatch):
    """A well-formed question is returned verbatim, with no LLM call at all."""
    called = []

    def boom(*a, **kw):
        called.append(1)
        raise AssertionError("chat() must not be called")

    monkeypatch.setattr("app.llm.chat", boom)
    q = "What is the maximum number of credits I can take in one semester?"
    rewrite_query.cache_clear()
    assert rewrite_query(q) == q
    assert not called


def test_rewrite_falls_back_to_original_when_llm_unavailable(monkeypatch):
    """No API key / provider down must degrade retrieval, not break the request."""
    def boom(*a, **kw):
        raise RuntimeError("LLM_API_KEY is not set")

    monkeypatch.setattr("app.llm.chat", boom)
    rewrite_query.cache_clear()
    assert rewrite_query("毕业需要多少学分") == "毕业需要多少学分"


def test_rewrite_rejects_a_rambling_response(monkeypatch):
    """A rewrite that turns into a paragraph is not trustworthy."""
    monkeypatch.setattr("app.llm.chat", lambda *a, **kw: "x" * 400)
    rewrite_query.cache_clear()
    assert rewrite_query("graduate") == "graduate"


def test_rewrite_rejects_multiline_response(monkeypatch):
    monkeypatch.setattr("app.llm.chat", lambda *a, **kw: "line one\nline two")
    rewrite_query.cache_clear()
    assert rewrite_query("graduate") == "graduate"


def test_rewrite_strips_quotes_the_model_may_add(monkeypatch):
    monkeypatch.setattr("app.llm.chat", lambda *a, **kw: '"What are the graduation requirements?"')
    rewrite_query.cache_clear()
    assert rewrite_query("graduate") == "What are the graduation requirements?"


def test_rewrite_result_is_cached(monkeypatch):
    """Same question twice must cost one LLM call, not two."""
    calls = []
    monkeypatch.setattr(
        "app.llm.chat",
        lambda *a, **kw: (calls.append(1), "graduation requirements")[1],
    )
    rewrite_query.cache_clear()
    rewrite_query("grad")
    rewrite_query("grad")
    assert len(calls) == 1


# ---------------------------------------------------- search() integration
def test_search_can_skip_normalisation(monkeypatch):
    """normalise=False must not touch the LLM — tests and A/B runs rely on it."""
    def boom(*a, **kw):
        raise AssertionError("rewrite must not run when normalise=False")

    monkeypatch.setattr(query_mod, "rewrite_query", boom)

    # Keep this test independent from a developer's pre-built persistent index.
    # A fresh CI checkout correctly has no data/chroma_db collection yet.
    import app.rag.index as index_mod

    class FakeCollection:
        def query(self, *, query_texts, n_results):
            assert query_texts == ["毕业需要多少学分"]
            assert n_results == 1
            return {
                "documents": [["Graduation\n\nComplete a minimum of 120 credits."]],
                "metadatas": [[{"source": "data/handbook.md", "heading": "Graduation"}]],
                "distances": [[0.3]],
            }

    monkeypatch.setattr(index_mod, "get_client", lambda: object())
    monkeypatch.setattr(index_mod, "get_collection", lambda _client: FakeCollection())

    hits = index_mod.search("毕业需要多少学分", k=1, normalise=False)
    assert len(hits) == 1
