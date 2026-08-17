"""Lesson 9 demo entrypoint.

The reusable implementation lives in ``app.agent.agent`` because the FastAPI
``/agent-chat`` route and the MCP server use the same agent and tools.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent.agent import (  # noqa: E402,F401
    BACKEND,
    SYSTEM_PROMPT,
    agent,
    chat,
    chat_with_trace,
    enrol,
    get_course,
    handbook_qa,
    memory,
)


if __name__ == "__main__":
    thread_id = "demo-session-1"
    for turn in [
        "Tell me about CS201.",
        "How many credits is it?",
        "Sign me up for it. My student id is 1.",
    ]:
        print(f"\nUSER: {turn}")
        result = chat_with_trace(turn, thread_id=thread_id)
        print(f"AGENT: {result['answer']}")
        print(f"TOOLS: {result['tool_calls']}")
