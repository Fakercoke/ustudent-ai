"""Regression tests for honest retrieval metrics."""
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
