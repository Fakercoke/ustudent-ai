"""Lesson 4 · POST /ask, /ask/v2, /ask/v3 — handbook assistant, 3 prompt versions.

Fill in the TODOs. Each version is a step up in prompt engineering:
    v1: system message defines role + handbook constraint
    v2: v1 + few-shot examples showing desired JSON shape
    v3: v2 + response_format={"type": "json_object"} + Pydantic parse

Also wire this router into app/main.py:
    from app.routes import ask
    app.include_router(ask.router, tags=["ask"])
"""
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.llm import chat
from app.ops.context import annotate_input

router = APIRouter()


# Load handbook once at import time. Kept small: pass the full text into
# the prompt. Lesson 6+ (RAG) will replace this with retrieval.
_HANDBOOK_PATH = Path("data/handbook.md")
HANDBOOK_TEXT = _HANDBOOK_PATH.read_text()


# ---- request / response models --------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskV1Response(BaseModel):
    answer: str


class AskV2Response(BaseModel):
    answer: str


class AskV3Response(BaseModel):
    answer: str
    citation: str


# ---- prompt builders (pure functions, unit-testable) ----------------------

def build_v1_messages(handbook: str, question: str) -> list[dict]:
    """招 1: system message defines the assistant's role + handbook constraint."""
    system = f"""You are a helpful student advisor at U+.
Answer ONLY using facts from the handbook below.
If the answer is not in the handbook, say "I don't know based on the handbook."

Handbook:
{handbook}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def build_v2_messages(handbook: str, question: str) -> list[dict]:
    """招 2: few-shot examples of the desired JSON format."""
    system = f"""You are a helpful student advisor at U+.
Answer using ONLY the handbook.
Return a JSON object: {{"answer": "...", "citation": "..."}}
If not in handbook: {{"answer": "unknown", "citation": ""}}

Handbook:
{handbook}

Examples:
User: How many credits do I need to graduate?
You: {{"answer": "120 credit points", "citation": "Undergraduate degree requires 120 credit points."}}

User: What's the capital of France?
You: {{"answer": "unknown", "citation": ""}}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def build_v3_messages(handbook: str, question: str) -> list[dict]:
    """招 3: keep few-shot; the endpoint will also use response_format."""
    # Reuse v2's prompt (schema + few-shot). The "force clean JSON" job is
    # handled by response_format={"type": "json_object"} at the endpoint level.
    return build_v2_messages(handbook, question)


# ---- endpoints ------------------------------------------------------------

@router.post("/ask", response_model=AskV1Response)
def ask_v1_endpoint(req: AskRequest) -> AskV1Response:
    annotate_input(req.question)
    messages = build_v1_messages(HANDBOOK_TEXT, req.question)
    answer = chat(messages, temperature=0)
    return AskV1Response(answer=answer)


@router.post("/ask/v2", response_model=AskV2Response)
def ask_v2_endpoint(req: AskRequest) -> AskV2Response:
    annotate_input(req.question)
    messages = build_v2_messages(HANDBOOK_TEXT, req.question)
    answer = chat(messages, temperature=0)
    return AskV2Response(answer=answer)


@router.post("/ask/v3", response_model=AskV3Response)
def ask_v3_endpoint(req: AskRequest) -> AskV3Response:
    annotate_input(req.question)
    messages = build_v3_messages(HANDBOOK_TEXT, req.question)
    raw = chat(
        messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return AskV3Response.model_validate_json(raw)
