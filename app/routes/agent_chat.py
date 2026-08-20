"""Lesson 9 · POST /agent-chat — multi-tool Agent with thread memory."""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.agent import chat_with_trace
from app.ops.context import annotate_input, annotate_tools

router = APIRouter()


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="One stable ID per conversation; different users must not share it.",
    )


class AgentChatResponse(BaseModel):
    answer: str
    thread_id: str
    tool_calls: list[dict]


@router.post("/agent-chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    annotate_input(req.message)
    result = chat_with_trace(req.message, thread_id=req.thread_id)
    annotate_tools(result["tool_calls"])
    return AgentChatResponse(
        answer=result["answer"],
        thread_id=req.thread_id,
        tool_calls=result["tool_calls"],
    )
