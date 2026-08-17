"""Lesson 7 starter — wire your lesson-5 index into an end-to-end RAG service.

You inherit a working Chroma index from lesson 5 (./chroma_db/). Your job is
the right half of the pipeline: retrieval-with-threshold + grounded prompt +
LLM call + sources passthrough.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Reuse the index lesson 5 built — adjust this path if yours is elsewhere.
ROOT = Path(__file__).resolve().parents[3]
DB_DIR = ROOT / "data/chroma_db"
COLLECTION = "ustudent_handbook"

# TODO 1: pick a distance threshold. Run a few queries in lesson 5 to see what
# distances on-topic vs off-topic give you. Reasonable starting point: 0.7.
DISTANCE_THRESHOLD = ...

# TODO 2: write the grounded prompt template. Must include:
#   - A clear role
#   - A negative constraint ("only use material below; say I don't know otherwise")
#   - {context} and {question} placeholders
RAG_PROMPT = """\
TODO
"""

FALLBACK_ANSWER = "I couldn't find anything in the student handbook that answers that. Please contact Academic Advising at advising@uplus.edu."


@dataclass
class RagResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    used_fallback: bool = False


def retrieve(question: str, k: int = 3) -> list[dict]:
    """Return top-K chunks as [{text, source, heading, distance}, ...]."""
    chroma = chromadb.PersistentClient(path=str(DB_DIR))
    col = chroma.get_collection(COLLECTION)
    res = col.query(query_texts=[question], n_results=k)

    # TODO 3: shape the result into a list of dicts with keys
    #   text, source (from metadata), heading (from metadata or ''), distance
    pass


def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()


def rag_answer(question: str, k: int = 3) -> RagResult:
    sources = retrieve(question, k=k)

    # TODO 4: if there are no sources, or the closest distance is above
    # DISTANCE_THRESHOLD, return RagResult(FALLBACK_ANSWER, sources, used_fallback=True).
    # Do NOT call the LLM in that case — burning tokens on hopeless queries is a bug.

    # TODO 5: build context by joining source texts with a separator and a
    # provenance tag like "[source · heading]" before each chunk.
    context = ...

    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = call_llm(prompt)

    # TODO 6: also flag used_fallback=True if the model itself said
    # something matching "i don't know". Don't overwrite answer — surface
    # the model's wording.
    pass


if __name__ == "__main__":
    # Manual smoke test before adding the FastAPI route.
    for q in [
        "How many credits do I need to graduate?",
        "Can a freshman take CS201?",
        "What AWS region does the system run in?",  # not in handbook -> fallback
    ]:
        print(f"\n=== Q: {q}")
        r = rag_answer(q)
        print(f"  fallback? {r.used_fallback}")
        print(f"  A: {r.answer}")
        print(f"  sources: {len(r.sources)} chunks, top distance {r.sources[0]['distance']:.3f}"
              if r.sources else "  sources: none")
