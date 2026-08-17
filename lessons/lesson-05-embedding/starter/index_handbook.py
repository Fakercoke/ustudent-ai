"""Lesson 5 starter — chunk the handbook, embed, store, retrieve.

Two parts:

PART 1 (toy):
  Write a cosine_similarity that takes two word-count dicts and returns a
  number in [0, 1]. Verify "drop the course" and "withdraw from the class"
  come out closer than "drop the course" and "today's weather".

PART 2 (real):
  Use Chroma to index data/handbook.md + data/faq.md + data/courses-catalog.md.
  Query for three real student questions and print the top-3 chunks.

By the end you'll have a runnable index living in ./chroma_db that lesson 7
will reuse for the actual RAG service.

Usage:
  python index_handbook.py demo       # Part 1 — hand-rolled cosine
  python index_handbook.py build      # Part 2 — build the index
  python index_handbook.py query      # Part 2 — the three required queries
  python index_handbook.py extra      # exercise step 4 — my own queries
  python index_handbook.py compare    # exercise step 3 — chunk size experiment
"""
from __future__ import annotations

import sys
from collections import Counter
from math import sqrt
from pathlib import Path

import chromadb


# ============================================================================
# PART 1 — toy bag-of-words cosine similarity
# ============================================================================

def bag_of_words(text: str) -> dict[str, int]:
    """Lower-case, split on whitespace, count tokens."""
    return dict(Counter(text.lower().split()))


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """cos(a, b) = dot(a, b) / (||a|| * ||b||). Returns 0 if either is empty."""
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def demo_part1():
    a = "drop the course before week two"
    b = "withdraw from the class by week two"
    c = "today's weather is wonderful"
    print(f"sim(a, b) = {cosine_similarity(bag_of_words(a), bag_of_words(b)):.3f}")
    print(f"sim(a, c) = {cosine_similarity(bag_of_words(a), bag_of_words(c)):.3f}")
    # If your implementation is right, a-b should be MUCH higher than a-c.


# ============================================================================
# PART 2 — real embeddings via Chroma
# ============================================================================

# Naive chunker. You can swap this for a markdown-heading-aware one later.
def chunk_text(text: str, size: int = 600, overlap: int | None = None) -> list[str]:
    """Slide a `size`-character window forward by (size - overlap) each step.

    `overlap` defaults to size // 6 rather than a fixed 100, so the chunk-size
    experiment stays fair: at the default size=600 that is exactly 100 (same as
    the original starter), but at size=100 a fixed 100 would make the step 0 and
    blow up with `ValueError: range() arg 3 must not be zero`.
    """
    if overlap is None:
        overlap = size // 6
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    step = size - overlap
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


CORPUS = ["data/handbook.md", "data/faq.md", "data/courses-catalog.md"]
ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(ROOT / "lessons/lesson-05-embedding/starter/chroma_db")

REQUIRED_QUERIES = [
    "How many credits do I need to graduate?",
    "Can I take CS101 and MATH101 together?",
    "How is GPA calculated?",
]

# Exercise step 4: three of my own. The last one has no answer in the corpus —
# its top-1 distance is what lesson 7's fallback threshold will be based on.
EXTRA_QUERIES = [
    "What happens if I fail a required course?",
    "How late can I withdraw without hurting my GPA?",
    "How do I get a parking permit?",  # <- deliberately unanswerable
]


def _collection(client, name: str = "handbook"):
    """Cosine space so distances land in [0, 2] instead of squared-L2."""
    return client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )


def build_index(size: int = 600, name: str = "handbook", quiet: bool = False) -> int:
    client = chromadb.PersistentClient(path=DB_PATH)

    # add() appends, it does not replace. Without this, re-running build would
    # stack duplicate copies of every chunk on top of the previous run.
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = _collection(client, name)

    total = 0
    for rel in CORPUS:
        chunks = chunk_text((ROOT / rel).read_text(), size=size)
        # IDs must be globally unique within the collection. "{path}::{i}"
        # keeps handbook chunk 3 from silently overwriting faq chunk 3.
        col.add(documents=chunks, ids=[f"{rel}::{i}" for i in range(len(chunks))])
        if not quiet:
            print(f"  {rel:28s} -> {len(chunks):4d} chunks")
        total += len(chunks)

    print(f"Indexed {total} chunks")
    return total


def _print_hits(col, q: str, n: int = 3) -> float:
    """Print top-n hits for one query. Returns the top-1 distance."""
    res = col.query(query_texts=[q], n_results=n)
    # Every value is nested one level deep because query_texts takes a list.
    dists, docs, ids = res["distances"][0], res["documents"][0], res["ids"][0]
    print(f"\nQ: {q}")
    for dist, doc, cid in zip(dists, docs, ids):
        preview = " ".join(doc.split())[:80]
        print(f"  [{dist:.3f}] {cid}")
        print(f"           {preview}...")
    return dists[0]


def query_index(queries: list[str] | None = None, name: str = "handbook"):
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(name)
    for q in queries or REQUIRED_QUERIES:
        _print_hits(col, q)


def compare_chunk_sizes(sizes: tuple[int, ...] = (100, 600, 3000)):
    """Exercise step 3: build the same corpus at several chunk sizes and
    print top-1 distance per query side by side."""
    counts: dict[int, int] = {}
    for size in sizes:
        print(f"\n=== building with size={size} ===")
        counts[size] = build_index(size=size, name=f"size_{size}", quiet=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    all_queries = REQUIRED_QUERIES + EXTRA_QUERIES

    print("\n" + "=" * 78)
    print("top-1 distance by chunk size (lower = better match)")
    print("=" * 78)
    header = "query".ljust(46) + "".join(f"{s:>10}" for s in sizes)
    print(header)
    print("-" * 78)
    for q in all_queries:
        row = q[:44].ljust(46)
        for size in sizes:
            col = client.get_collection(f"size_{size}")
            res = col.query(query_texts=[q], n_results=1)
            row += f"{res['distances'][0][0]:>10.3f}"
        print(row)
    print("-" * 78)
    print("chunks".ljust(46) + "".join(f"{counts[s]:>10}" for s in sizes))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "demo":
        demo_part1()
    elif cmd == "build":
        build_index()
        print("Index built.")
    elif cmd == "query":
        query_index()
    elif cmd == "extra":
        query_index(EXTRA_QUERIES)
    elif cmd == "compare":
        compare_chunk_sizes()
    else:
        print("Usage: python index_handbook.py [demo|build|query|extra|compare]")
