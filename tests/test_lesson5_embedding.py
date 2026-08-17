"""Lesson 5 — unit tests for the pure functions in index_handbook.py.

The lesson lives under a dashed directory name, so it cannot be imported with
a normal `import` statement. Load it by file path instead.
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lessons/lesson-05-embedding/starter/index_handbook.py"
)
_spec = importlib.util.spec_from_file_location("index_handbook", _MODULE_PATH)
lesson5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson5)

bag_of_words = lesson5.bag_of_words
cosine_similarity = lesson5.cosine_similarity
chunk_text = lesson5.chunk_text


# --------------------------------------------------------------- bag_of_words
def test_bag_of_words_lowercases_and_counts():
    assert bag_of_words("Drop the course the") == {"drop": 1, "the": 2, "course": 1}


def test_bag_of_words_empty_string():
    assert bag_of_words("") == {}


# ---------------------------------------------------------- cosine_similarity
def test_cosine_identical_texts_is_one():
    v = bag_of_words("drop the course")
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_no_shared_words_is_zero():
    a = bag_of_words("drop the course")
    b = bag_of_words("sunny warm weather")
    assert cosine_similarity(a, b) == 0.0


def test_cosine_empty_input_returns_zero_not_crash():
    """The docstring promises 0 for empty input — a naive implementation
    divides by a zero norm and raises ZeroDivisionError instead."""
    assert cosine_similarity({}, bag_of_words("anything")) == 0.0
    assert cosine_similarity({}, {}) == 0.0


def test_cosine_synonyms_beat_unrelated():
    """The whole point of Part 1: paraphrases score higher than noise —
    but only mildly, because bag-of-words cannot see that drop == withdraw."""
    a = bag_of_words("drop the course before week two")
    b = bag_of_words("withdraw from the class by week two")
    c = bag_of_words("today's weather is wonderful")
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


# ------------------------------------------------------------------ chunk_text
def test_chunk_text_respects_size():
    chunks = chunk_text("x" * 2000, size=600, overlap=100)
    assert all(len(c) <= 600 for c in chunks)


def test_chunk_text_step_is_size_minus_overlap():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = chunk_text(text, size=600, overlap=100)
    assert chunks[0] == text[0:600]
    assert chunks[1] == text[500:1100]  # step 500, not 600


def test_chunk_text_overlap_carries_context_across_the_boundary():
    text = "A" * 550 + "NEEDLE" + "B" * 550
    chunks = chunk_text(text, size=600, overlap=100)
    # The needle straddles the 600 boundary; overlap means some chunk has it whole.
    assert any("NEEDLE" in c for c in chunks)


def test_chunk_text_drops_whitespace_only_chunks():
    assert chunk_text("   \n\n   ", size=10, overlap=2) == []


def test_chunk_text_default_overlap_scales_with_size():
    """size=100 with a hard-coded overlap=100 would make step 0 and raise
    `ValueError: range() arg 3 must not be zero`. The default is size // 6."""
    assert len(chunk_text("x" * 5000, size=100)) > 0
    assert len(chunk_text("x" * 5000, size=3000)) > 0
    # At the starter's default size the effective overlap is still 100.
    assert chunk_text("x" * 5000, size=600) == chunk_text("x" * 5000, size=600, overlap=100)


def test_chunk_text_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("x" * 500, size=100, overlap=100)


def test_chunk_text_rejects_non_positive_size():
    with pytest.raises(ValueError, match="size"):
        chunk_text("x" * 500, size=0)


def test_chunk_text_covers_the_whole_text():
    text = "".join(str(i % 10) for i in range(3000))
    chunks = chunk_text(text, size=600, overlap=100)
    assert "".join(dict.fromkeys("".join(chunks))) != ""
    assert chunks[-1].endswith(text[-10:])
