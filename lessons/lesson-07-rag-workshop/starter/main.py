"""Lesson 7 starter — expose your RAG as a FastAPI endpoint."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import rag_answer

app = FastAPI(title="ustudent RAG service")


@app.get("/health")
def health():
    return {"status": "ok"}


class RagAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=10)


# TODO: define a RagAskResponse model with:
#   answer: str
#   sources: list of {text, source, heading, distance}
#   used_fallback: bool


@app.post("/rag-ask")
def rag_ask(req: RagAskRequest):
    # TODO: call rag_answer, return the result shaped as RagAskResponse.
    pass
