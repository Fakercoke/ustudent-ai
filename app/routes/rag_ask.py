"""Lesson 7 · POST /rag-ask — 作品一的对外接口.

Answers handbook questions from the vector index, and returns the retrieved
chunks alongside the answer so callers can verify it — and so failures can be
diagnosed without guessing.

`sources` is not decoration. When an answer is wrong there are three different
causes needing three different fixes, and only `sources` + `used_fallback`
together tell them apart:

    sources 里没有正确答案                -> 检索问题 -> 调 chunk / top-k / 切法
    sources 里有，但模型答错了             -> 生成问题 -> 调 prompt
    sources 里有，但 used_fallback=true   -> 阈值问题 -> 放宽阈值
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.pipeline import (
    DISTANCE_THRESHOLD,
    MAX_QUESTION_CHARS,
    TOP_K,
    rag_answer,
)

router = APIRouter()


class RagAskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=f"问题文本，最长 {MAX_QUESTION_CHARS} 字符。超长部分对句向量模型是噪音，"
                    "也是最省事的一种攻击面",
        examples=["How many credits do I need to graduate?"],
    )
    k: int = Field(TOP_K, ge=1, le=10, description="要检索几块。默认 3，实测 k=1 答对率仅 50%")


class Source(BaseModel):
    text: str
    source: str = Field(..., description="来自哪个文件")
    heading: str = Field(..., description="来自哪一节")
    distance: float = Field(..., description=f"余弦距离，越小越像。超过 {DISTANCE_THRESHOLD} 触发兜底")


class RagAskResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_fallback: bool = Field(
        ..., description="true 表示没有作答：要么距离超阈值，要么模型判定资料不足"
    )
    blocked: bool = Field(
        False,
        description="true 表示检索到的手册内容含疑似指令注入，请求在调用模型前被中止",
    )
    degraded: bool = Field(
        False,
        description="true 表示生成模型不可用，answer 里是检索到的手册原文而非 AI 总结",
    )


@router.post("/rag-ask", response_model=RagAskResponse)
def rag_ask(req: RagAskRequest) -> RagAskResponse:
    result = rag_answer(req.question, k=req.k)
    return RagAskResponse(
        answer=result.answer,
        sources=[Source(**h) for h in result.sources],
        used_fallback=result.used_fallback,
        degraded=result.degraded,
        blocked=result.blocked,
    )
