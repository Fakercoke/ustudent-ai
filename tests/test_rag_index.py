"""Tests for app/rag/index.py — the production chunking + indexing module."""
import pytest

from app.rag import index as rag_index
from app.rag.index import (
    CHUNK_SIZE,
    chunk_document,
    split_by_heading,
    split_windows,
)

SAMPLE = """# Student Handbook

Intro paragraph.

## 1. Credits

You need 120 credits to graduate.

## 2. Refunds

Full refund before end of Week 2.
"""


# ------------------------------------------------------------- split_windows
def test_split_windows_step_is_size_minus_overlap():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = split_windows(text, 600, 100)
    assert chunks[0] == text[0:600]
    assert chunks[1] == text[500:1100]  # step 500, not 600


def test_split_windows_rejects_overlap_not_smaller_than_size():
    """size=100 with overlap=100 gives step 0 -> range() would raise a
    confusing 'arg 3 must not be zero'. Fail with a readable message instead."""
    with pytest.raises(ValueError, match="overlap"):
        split_windows("x" * 500, 100, 100)


def test_split_windows_rejects_non_positive_size():
    with pytest.raises(ValueError, match="size"):
        split_windows("x" * 500, 0, 0)


def test_split_windows_drops_whitespace_only_pieces():
    assert split_windows("   \n\n   ", 10, 2) == []


# ---------------------------------------------------------- split_by_heading
def test_split_by_heading_splits_on_every_heading():
    sections = list(split_by_heading(SAMPLE))
    assert [h for h, _, _ in sections] == [
        "Student Handbook", "1. Credits", "2. Refunds"
    ]


def test_split_by_heading_records_level():
    levels = {h: lvl for h, lvl, _ in split_by_heading(SAMPLE)}
    assert levels["Student Handbook"] == 1
    assert levels["1. Credits"] == 2


def test_split_by_heading_keeps_a_fact_intact():
    """The whole point of heading-aware chunking: a sentence that a fixed
    window would cut in half stays whole inside its section."""
    bodies = [b for _, _, b in split_by_heading(SAMPLE)]
    assert any("Full refund before end of Week 2." in b for b in bodies)


# ------------------------------------------------------------ chunk_document
def test_chunk_document_prefixes_heading_into_the_text():
    """A chunk reading 'You need 120 credits' is ambiguous on its own —
    the heading has to travel with it into the embedding."""
    chunks = chunk_document(SAMPLE, "data/handbook.md")
    credits = next(c for c in chunks if "120 credits" in c["text"])
    assert credits["text"].startswith("1. Credits")
    assert credits["heading"] == "1. Credits"


def test_chunk_document_sets_source_metadata():
    chunks = chunk_document(SAMPLE, "data/handbook.md")
    assert all(c["source"] == "data/handbook.md" for c in chunks)


def test_chunk_document_never_exceeds_chunk_size():
    long_section = "# Big\n\n" + ("word " * 2000)
    chunks = chunk_document(long_section, "x.md")
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)
    assert len(chunks) > 1  # the long section really was split


def test_chunk_document_short_section_stays_one_chunk():
    chunks = chunk_document("# Tiny\n\nshort body.", "x.md")
    assert len(chunks) == 1


# ----------------------------------------------------- index + search (slow)
@pytest.fixture(scope="module")
def indexed_client(tmp_path_factory):
    """Build a throwaway index in a temp dir so tests never touch data/chroma_db."""
    client = rag_index.get_client(tmp_path_factory.mktemp("chroma"))
    n = rag_index.index_documents(
        client,
        {
            "data/handbook.md": SAMPLE,
            "data/faq.md": "# FAQ\n\n## Q1. Parking?\n\nNot covered here.",
        },
    )
    return client, n


def test_index_documents_returns_chunk_count(indexed_client):
    client, n = indexed_client
    assert n == rag_index.get_collection(client).count()
    assert n > 0


def test_index_ids_are_unique_across_files(indexed_client):
    """str(i) alone would let handbook chunk 3 silently overwrite faq chunk 3."""
    client, n = indexed_client
    ids = rag_index.get_collection(client).get()["ids"]
    assert len(ids) == len(set(ids)) == n
    assert any(i.startswith("data/handbook.md::") for i in ids)
    assert any(i.startswith("data/faq.md::") for i in ids)


def test_index_documents_is_idempotent(indexed_client):
    """add() appends — rebuilding must not stack duplicates."""
    client, n = indexed_client
    again = rag_index.index_documents(
        client,
        {
            "data/handbook.md": SAMPLE,
            "data/faq.md": "# FAQ\n\n## Q1. Parking?\n\nNot covered here.",
        },
    )
    assert again == n
    assert rag_index.get_collection(client).count() == n
