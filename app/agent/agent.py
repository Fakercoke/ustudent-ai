"""Lesson 9: a three-tool LangGraph agent with per-thread memory."""
from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.ops.context import RequestTrace, current_trace, record_llm_call
from app.rag.pipeline import rag_answer

load_dotenv()

BACKEND = os.environ.get("USTUDENT_BACKEND_URL", "http://localhost:8080")
HTTP_TIMEOUT = 10.0


def _fetch_courses() -> list[dict[str, Any]]:
    """Fetch and validate the backend's course-list envelope."""
    response = httpx.get(f"{BACKEND}/api/courses", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("success") is not True:
        message = payload.get("message", "invalid backend response") if isinstance(payload, dict) else "invalid backend response"
        raise ValueError(message)
    courses = payload.get("data")
    if not isinstance(courses, list):
        raise ValueError("backend response does not contain a course list")
    return courses


def _find_course(course_code: str) -> dict[str, Any] | None:
    wanted = course_code.strip().upper()
    return next(
        (course for course in _fetch_courses()
         if str(course.get("course_code", "")).upper() == wanted),
        None,
    )


def _backend_error(exc: Exception) -> str:
    """Turn transport/backend failures into a safe tool result for the agent."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            payload = exc.response.json()
            return str(payload.get("message") or payload)
        except (ValueError, AttributeError):
            return exc.response.text or str(exc)
    return str(exc)


@tool
def handbook_qa(question: str) -> str:
    """Answer policy and handbook questions about GPA, drops, prerequisites,
    refunds, graduation, and other university rules using the grounded RAG index.
    """
    result = rag_answer(question)
    if result.used_fallback:
        return f"(no handbook answer) {result.answer}"
    return result.answer


@tool
def get_course(course_code: str) -> str:
    """Look up verified backend details for one course code such as CS101 or
    MATH201. Returns its name, description, credits, enrolment and teacher.
    """
    try:
        course = _find_course(course_code)
    except (httpx.HTTPError, ValueError) as exc:
        return f"Course lookup failed: {_backend_error(exc)}"
    if course is None:
        return f"No course found with code {course_code.strip().upper()}."

    teacher = course.get("teacher") or {}
    teacher_name = teacher.get("full_name", "unassigned") if isinstance(teacher, dict) else str(teacher)
    return (
        f"{course['course_code']} — {course.get('course_name', 'Unnamed course')}; "
        f"description: {course.get('description', 'not provided')}; "
        f"{course.get('credits', 'unknown')} credits; "
        f"enrolment {course.get('current_enrollments', 'unknown')}/"
        f"{course.get('max_students', 'unknown')}; teacher: {teacher_name}."
    )


@tool
def enrol(student_id: int, course_code: str) -> str:
    """Enrol a student in a course by course code. This changes real backend
    data, so call it only when the user explicitly asks to enrol and supplies a
    student ID. Returns the backend's success or rejection reason.
    """
    try:
        course = _find_course(course_code)
        if course is None:
            return f"Enrolment rejected: no course found with code {course_code.strip().upper()}."

        response = httpx.post(
            f"{BACKEND}/api/courses/{course['id']}/enroll",
            params={"studentId": student_id},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return f"Enrolment rejected: {_backend_error(exc)}"

    if not isinstance(payload, dict):
        return "Enrolment rejected: backend returned an invalid response."
    message = str(payload.get("message") or "backend did not provide a reason")
    if payload.get("success") is not True:
        return f"Enrolment rejected: {message}"
    return f"Successfully enrolled student {student_id} in {course['course_code']}: {message}"


SYSTEM_PROMPT = """\
You are the ustudent course-enrolment assistant.

Tool choice rules:
* Policy or handbook questions (GPA, drop deadlines, prerequisites, refunds,
  graduation, etc.) -> handbook_qa.
* Course facts such as description, credits, capacity or teacher -> get_course.
* An explicit request to sign up or enrol -> enrol. Resolve pronouns such as
  "it" or "that course" to the most recent course discussed in this thread.

Use previous messages and tool results in this thread before calling a tool
again. Treat tool results as the only source of truth and never invent course or
policy information. Never claim enrolment succeeded unless enrol returned a
success message. If a tool returns an error or rejection, report it honestly.
"""

_settings = get_settings()
_base_url = _settings.llm_base_url
_extra_body = (
    {"thinking": {"type": "disabled"}}
    if "api.deepseek.com" in _base_url
    else None
)

llm = ChatOpenAI(
    # Keep import/health/tests available before a key is configured.  A real
    # Agent call still fails honestly at the provider boundary.
    api_key=_settings.llm_api_key or "not-configured",
    base_url=_base_url,
    model=_settings.llm_model,
    temperature=0.1,
    extra_body=_extra_body,
)

# Module-level memory is shared by calls to chat(). Different thread IDs keep
# conversations isolated from one another.
memory = MemorySaver()
agent = create_react_agent(
    llm,
    tools=[handbook_qa, get_course, enrol],
    state_modifier=SYSTEM_PROMPT,
    checkpointer=memory,
)


class OpsLLMCallback(BaseCallbackHandler):
    """Record each Agent model call immediately, including failed turns.

    Reading usage only after `agent.invoke()` returns loses both the error and
    any earlier successful ReAct step when a later call fails.  A per-request
    callback receives every model completion/error at the correct boundary.
    """

    def __init__(self, model: str, trace: RequestTrace | None = None) -> None:
        self.model = model
        self.trace = trace or current_trace()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        output = response.llm_output or {}
        usage = output.get("token_usage", {}) or {}
        model = str(output.get("model_name") or self.model)

        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        cached = int(
            usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0)) or 0
        )

        # Some provider adapters expose usage on the returned AIMessage rather
        # than LLMResult.llm_output. Sum it only when the aggregate is absent.
        if not (prompt or completion or total):
            for batch in response.generations:
                for generation in batch:
                    metadata = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    prompt += int(metadata.get("input_tokens", 0) or 0)
                    completion += int(metadata.get("output_tokens", 0) or 0)
                    total += int(metadata.get("total_tokens", 0) or 0)
                    input_details = metadata.get("input_token_details", {}) or {}
                    cached += int(input_details.get("cache_read", 0) or 0)

        record_llm_call(
            model=model,
            prompt_tokens=prompt,
            cached_prompt_tokens=cached,
            completion_tokens=completion,
            total_tokens=total,
            trace=self.trace,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        record_llm_call(model=self.model, success=False, trace=self.trace)


def chat_with_trace(message: str, thread_id: str) -> dict[str, Any]:
    """Run one turn and expose only that turn's tool calls for diagnostics."""
    message = message.strip()
    thread_id = thread_id.strip()
    if not message:
        raise ValueError("message must not be empty")
    if not thread_id:
        raise ValueError("thread_id must not be empty")

    result = agent.invoke(
        {"messages": [("user", message)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 15,
            "callbacks": [OpsLLMCallback(_settings.llm_model)],
        },
    )
    messages = result["messages"]
    last_user = max(
        index for index, item in enumerate(messages)
        if isinstance(item, HumanMessage)
    )
    current_turn = messages[last_user + 1:]
    tool_results = {
        item.tool_call_id: str(item.content)
        for item in current_turn
        if isinstance(item, ToolMessage)
    }
    tool_calls = []
    for item in current_turn:
        for call in getattr(item, "tool_calls", []) or []:
            tool_calls.append({
                "name": call.get("name", "unknown"),
                "args": call.get("args", {}),
                "result": tool_results.get(call.get("id", ""), ""),
            })
    return {
        "answer": str(messages[-1].content),
        "tool_calls": tool_calls,
    }


def chat(message: str, thread_id: str) -> str:
    """Run one conversational turn and return only the assistant text."""
    return str(chat_with_trace(message, thread_id)["answer"])
