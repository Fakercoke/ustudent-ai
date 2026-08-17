"""Lesson 3 · TDD-driven tests for POST /can-graduate.

Follow the ORDER: run pytest after each step and watch it go from RED to GREEN.

    Step 2 (in class):  uncomment TODO 1, run pytest — RED (404)
                        implement the endpoint in app/routes/can_graduate.py,
                        wire it into app/main.py — GREEN
    Step 4 (homework):  uncomment TODO 2..5, all should GREEN

Do NOT delete the TODO comments — they're your checklist.
"""
from fastapi.testclient import TestClient
from app.main import app
from app.routes.can_graduate import GraduationRequest

client = TestClient(app)


# TODO 1 (in class) — the happy path
def test_120_credits_and_gpa_2_can_graduate():
    r = client.post("/can-graduate", json={"credits": 120, "gpa": 2.0})
    assert r.status_code == 200
    assert r.json() == {"can_graduate": True}


# TODO 2 — not enough credits
def test_not_enough_credits():
    r = client.post("/can-graduate", json={"credits": 100, "gpa": 3.5})
    assert r.status_code == 200
    assert r.json() == {"can_graduate": False}


# TODO 3 — gpa too low
def test_gpa_too_low():
    r = client.post("/can-graduate", json={"credits": 150, "gpa": 1.8})
    assert r.status_code == 200
    assert r.json() == {"can_graduate": False}


# TODO 4 — boundary case: exactly on the requirement
def test_exactly_120_and_2_passes():
    r = client.post("/can-graduate", json={"credits": 120, "gpa": 2.0})
    assert r.status_code == 200
    assert r.json() == {"can_graduate": True}


# TODO 5 — boundary case: one below credit requirement
def test_119_credits_fails():
    r = client.post("/can-graduate", json={"credits": 119, "gpa": 4.0})
    assert r.status_code == 200
    assert r.json() == {"can_graduate": False}


# BONUS — direct unit tests on the method itself (no HTTP layer).
# These call GraduationRequest.is_eligible() straight, so they test the
# business rule in isolation rather than the web endpoint.
def test_is_eligible_pass():
    assert GraduationRequest(credits=120, gpa=2.0).is_eligible() is True


def test_is_eligible_credits_too_low():
    assert GraduationRequest(credits=119, gpa=4.0).is_eligible() is False


def test_is_eligible_gpa_too_low():
    assert GraduationRequest(credits=150, gpa=1.8).is_eligible() is False


def test_is_eligible_both_fail():
    # the 4th truth-table case: neither condition met
    assert GraduationRequest(credits=100, gpa=1.5).is_eligible() is False
