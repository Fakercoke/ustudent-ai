"""Build and query the handbook vector index.

The interface (`get_client`, `index_documents`) is the one
`scripts/build_index.py` expects. `search()` is the retrieval half that
lesson 7's pipeline builds on.

    python scripts/build_index.py            # 建索引
    python -m app.rag.index search "问题"     # 查一下
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator, TypedDict

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data/chroma_db"
COLLECTION = "ustudent_handbook"

CORPUS = ["data/handbook.md", "data/faq.md", "data/courses-catalog.md"]

# Chosen from the lesson-5 experiment: size 100 truncates answers mid-sentence
# and only satisfies 3 of 6 golden questions; size 3000 costs 4.5x the prompt
# tokens for the same accuracy. See lessons/lesson-05-embedding/starter/report.md.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Chroma otherwise picks every locally available ONNX provider.  On macOS that
# can select CoreML, while Linux/Docker uses CPU, making tests and deployment
# behave differently.  The model itself is unchanged; fixing the execution
# provider makes local, CI and cloud runs reproducible.
_EMBEDDING_FUNCTION = ONNXMiniLM_L6_V2(
    preferred_providers=["CPUExecutionProvider"]
)


class Chunk(TypedDict):
    text: str
    source: str
    heading: str
    level: int


class Hit(TypedDict):
    text: str
    source: str
    heading: str
    distance: float


# ---------------------------------------------------------------- chunking

def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-width sliding window — the fallback for sections longer than `size`."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    step = size - overlap
    return [p for i in range(0, len(text), step) if (p := text[i:i + size].strip())]


def split_by_heading(markdown: str) -> Iterator[tuple[str, int, str]]:
    """Yield (heading, level, body) for each markdown section.

    Keeping a section together is what stops a fact from being cut in half:
    "End of Week 2 ... full refund and no academic record" lives inside one
    heading, so it survives as a unit no matter where a character window
    would have landed.
    """
    heading, level, buf = "", 0, []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            if buf and "".join(buf).strip():
                yield heading, level, "\n".join(buf).strip()
            heading, level, buf = m.group(2).strip(), len(m.group(1)), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        yield heading, level, "\n".join(buf).strip()


def chunk_document(markdown: str, source: str) -> list[Chunk]:
    """Heading-aware chunking: one chunk per section, windowed only if too long."""
    chunks: list[Chunk] = []
    for heading, level, body in split_by_heading(markdown):
        # Prefix the heading so the embedded text carries its own context —
        # a chunk reading "You need 120 credits" is ambiguous on its own.
        titled = f"{heading}\n\n{body}" if heading else body
        pieces = (
            [titled] if len(titled) <= CHUNK_SIZE
            else split_windows(titled, CHUNK_SIZE, CHUNK_OVERLAP)
        )
        for piece in pieces:
            chunks.append(
                {"text": piece, "source": source, "heading": heading, "level": level}
            )
    return chunks


# ------------------------------------------------------------------- index

def get_client(path: Path | str = DB_DIR):
    """Persistent Chroma client. Ephemeral clients lose the index on exit."""
    return chromadb.PersistentClient(path=str(path))


def get_collection(client):
    """cosine keeps distances in [0, 2]; Chroma's default squared-L2 is harder to read."""
    return client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_EMBEDDING_FUNCTION,
    )


def index_documents(client, documents: dict[str, str]) -> int:
    """Chunk every document and (re)build the collection. Returns chunk count."""
    # add() appends; without this a rebuild stacks duplicates on the old data.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection(client)

    total = 0
    for source, text in documents.items():
        chunks = chunk_document(text, source)
        col.add(
            documents=[c["text"] for c in chunks],
            # IDs must be unique across the whole collection, not per file —
            # "3" from handbook.md would otherwise silently overwrite "3" from faq.md.
            ids=[f"{source}::{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": c["source"], "heading": c["heading"], "level": c["level"]}
                for c in chunks
            ],
        )
        total += len(chunks)
    return total


def build_index(verbose: bool = True) -> int:
    """Convenience wrapper: read the corpus off disk and index it."""
    documents = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in CORPUS}
    total = index_documents(get_client(), documents)
    if verbose:
        print(f"Indexed {total} chunks into '{COLLECTION}' at {DB_DIR}")
    return total


# ---------------------------------------------------------------- retrieval

def search(question: str, k: int = 3, *, normalise: bool = True) -> list[Hit]:
    """Top-k chunks for `question`, shaped the way lesson 7's retrieve() wants.

    `normalise=True` runs the query through `app.rag.query.rewrite_query` first:
    fixes typos, expands one-word queries, translates non-English. The rewrite
    affects **retrieval only** — callers still answer the user's original
    wording. Pass `normalise=False` to search the raw text (used by tests and
    by the A/B comparison in the design doc).
    """
    if normalise:
        from app.rag.query import rewrite_query

        question = rewrite_query(question)

    col = get_collection(get_client())
    res = col.query(query_texts=[question], n_results=k)
    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "heading": meta.get("heading", ""),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build_index()
    elif cmd == "search" and len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\nQ: {q}")
        for h in search(q):
            preview = " ".join(h["text"].split())[:70]
            print(f"  [{h['distance']:.3f}] {h['source']} § {h['heading']}")
            print(f"           {preview}...")
    else:
        print("Usage: python -m app.rag.index [build|search <question>]")
