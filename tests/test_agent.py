"""Contract tests for the Lesson 9 tools and /agent-chat surface."""
from types import SimpleNamespace

import httpx

import app.agent.agent as agent_module
import app.routes.agent_chat as agent_route


COURSE = {
    "id": 2,
    "course_code": "CS201",
    "course_name": "Data Structures and Algorithms",
    "description": "Advanced programming with data structures",
    "credits": 4,
    "max_students": 25,
    "current_enrollments": 7,
    "teacher": {"full_name": "Dr. Wilson"},
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)
        self.request = httpx.Request("POST", "http://backend.test")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "backend error", request=self.request, response=self
            )


def test_get_course_formats_verified_backend_fields(monkeypatch):
    monkeypatch.setattr(agent_module, "_fetch_courses", lambda: [COURSE])

    result = agent_module.get_course.invoke({"course_code": "cs201"})

    assert "Data Structures and Algorithms" in result
    assert "4 credits" in result
    assert "enrolment 7/25" in result
    assert "Dr. Wilson" in result


def test_get_course_does_not_invent_missing_course(monkeypatch):
    monkeypatch.setattr(agent_module, "_fetch_courses", lambda: [COURSE])
    result = agent_module.get_course.invoke({"course_code": "CS999"})
    assert result == "No course found with code CS999."


def test_enrol_requires_explicit_backend_success(monkeypatch):
    monkeypatch.setattr(agent_module, "_find_course", lambda code: COURSE)
    monkeypatch.setattr(
        agent_module.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse({
            "success": False,
            "message": "Missing prerequisite: CS101",
        }),
    )

    result = agent_module.enrol.invoke({"student_id": 1, "course_code": "CS201"})

    assert result == "Enrolment rejected: Missing prerequisite: CS101"
    assert "Successfully" not in result


def test_enrol_reports_confirmed_success(monkeypatch):
    monkeypatch.setattr(agent_module, "_find_course", lambda code: COURSE)
    monkeypatch.setattr(
        agent_module.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse({
            "success": True,
            "message": "Enrollment successful",
        }),
    )

    result = agent_module.enrol.invoke({"student_id": 1, "course_code": "CS201"})

    assert result.startswith("Successfully enrolled student 1 in CS201")


def test_handbook_tool_marks_rag_fallback(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "rag_answer",
        lambda question: SimpleNamespace(answer="Contact advising.", used_fallback=True),
    )
    result = agent_module.handbook_qa.invoke({"question": "unknown policy"})
    assert result == "(no handbook answer) Contact advising."


def test_agent_chat_endpoint_returns_thread_and_trace(client, monkeypatch):
    monkeypatch.setattr(
        agent_route,
        "chat_with_trace",
        lambda message, thread_id: {
            "answer": "CS201 has 4 credits.",
            "tool_calls": [{"name": "get_course", "args": {"course_code": "CS201"}, "result": "4 credits"}],
        },
    )

    response = client.post(
        "/agent-chat",
        json={"message": "Tell me about CS201", "thread_id": "thread-a"},
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == "thread-a"
    assert response.json()["tool_calls"][0]["name"] == "get_course"


def test_agent_chat_rejects_empty_ids_and_messages(client):
    response = client.post("/agent-chat", json={"message": "", "thread_id": ""})
    assert response.status_code == 422

