"""Regression tests for honest retrieval metrics."""
import json
from types import SimpleNamespace

import scripts.eval_rag as eval_rag
from scripts.eval_rag import retrieval_contains_expected_fact


ITEM = {
    "expected_heading_any_of": ["CS101 — Introduction to Computer Science"],
}


def test_identifier_mention_in_wrong_section_is_not_a_retrieval_hit():
    sources = [{
        "heading": "CS201 — Data Structures and Algorithms",
        "text": "Prerequisite: CS101",
    }]
    assert not retrieval_contains_expected_fact(ITEM, "CS101", sources)


def test_identifier_and_fact_in_expected_section_is_a_retrieval_hit():
    sources = [{
        "heading": "CS101 — Introduction to Computer Science",
        "text": "CS101 — Introduction to Computer Science\nCredits: 3",
    }]
    assert retrieval_contains_expected_fact(ITEM, "CS101", sources)


def test_eval_forces_deterministic_generation(tmp_path, monkeypatch):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"items": [{
        "id": "probe-1",
        "q": "not covered",
        "in_handbook": False,
    }]}))
    monkeypatch.setitem(eval_rag.SETS, "probe", probe)
    temperatures: list[float] = []

    def fake_rag_answer(question: str, *, temperature: float):
        temperatures.append(temperature)
        return SimpleNamespace(used_fallback=True, sources=[], answer="fallback")

    monkeypatch.setattr(eval_rag, "rag_answer", fake_rag_answer)
    metrics = eval_rag.run("probe", verbose=False)

    assert temperatures == [0.0]
    assert metrics["refusal_accuracy"] == 1.0
