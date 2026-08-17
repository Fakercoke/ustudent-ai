"""Lesson 8 starter — same agent, written with LangGraph.

Side-by-side with react_loop.py (your hand-rolled version), this is what the
framework saves you. Should be ~15 lines once filled in.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """\
You are the ustudent course-enrolment assistant.
Use get_course for course details and policy_qa for handbook or policy questions.
For each requested fact, call the relevant tool once. Treat tool results as the
only source of truth: never add details that the tool did not return. After you
have the needed tool result, answer the user instead of calling the same tool
again. If one question asks for two different facts, call both relevant tools.
"""


# Same fake data as react_loop.py — apples to apples.
FAKE_COURSES = {
    "CS101": "CS101 — Introduction to Computer Science, 3 credits, Dr. Wilson.",
    "CS201": "CS201 — Data Structures and Algorithms, 4 credits, Dr. Wilson.",
    "MATH101": "MATH101 — Calculus I, 4 credits, Dr. Brown.",
}


@tool
def get_course(course_code: str) -> str:
    """Look up details for a single course by code (e.g. CS101)."""
    return FAKE_COURSES.get(course_code.upper(), f"No course found with code {course_code}.")


@tool
def policy_qa(question: str) -> str:
    """Answer a policy question about graduation, drops, GPA, etc."""
    lowered = question.lower()
    if "graduate" in lowered or "graduation" in lowered:
        return "You need 120 credits and GPA >= 2.0."
    if "drop" in lowered:
        return "Drop with full refund: end of Week 2."
    return "I don't know based on the handbook."


# Build the OpenAI-compatible chat model.
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.0,
)

# Let LangGraph own the ReAct loop and tool dispatch.
agent = create_react_agent(
    llm,
    tools=[get_course, policy_qa],
    state_modifier=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    for q in [
        "What is CS201 about?",
        "How many credits do I need to graduate?",
        "Look up CS201 and then tell me when I can drop courses.",
    ]:
        print(f"\n=== Q: {q}")
        # Invoke one independent conversation and print its final message.
        result = agent.invoke({"messages": [("user", q)]})
        print(f"  A: {result['messages'][-1].content}")
