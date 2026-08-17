"""FastAPI entrypoint for the ustudent-ai service.

Every new endpoint you write goes into a module under `app/routes/`
and is wired in below with `app.include_router(...)`.
"""
from fastapi import FastAPI

from app.routes import agent_chat, ask, can_graduate, echo, health, home, rag_ask

app = FastAPI(
    title="ustudent AI service",
    version="0.1.0",
)

app.include_router(home.router)
app.include_router(health.router, tags=["health"])
app.include_router(echo.router, tags=["echo"])
app.include_router(can_graduate.router, tags=["lesson-3"])
app.include_router(ask.router, tags=["ask"])
app.include_router(rag_ask.router, tags=["lesson-7 · 作品一 RAG"])
app.include_router(agent_chat.router, tags=["lesson-9 · 作品二 Agent"])
