"""Deprecated shim — the real implementation moved to `app/rag/index.py`.

`scripts/build_index.py` and lesson 9's agent both expect the code to live
under `app/rag/`, so that is now the canonical location. This module only
re-exports, so older imports keep working.

Prefer:
    from app.rag.index import search, build_index
"""
from app.rag.index import (  # noqa: F401
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    CORPUS,
    DB_DIR,
    ROOT,
    Chunk,
    Hit,
    build_index,
    chunk_document,
    get_client,
    get_collection,
    index_documents,
    search,
    split_by_heading,
    split_windows,
)

__all__ = [
    "CHUNK_OVERLAP", "CHUNK_SIZE", "COLLECTION", "CORPUS", "DB_DIR", "ROOT",
    "Chunk", "Hit", "build_index", "chunk_document", "get_client",
    "get_collection", "index_documents", "search", "split_by_heading",
    "split_windows",
]
