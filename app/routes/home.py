"""GET / — the page a recruiter or reviewer lands on.

`/docs` is generated for people who already know the service exists and what it
does. This is the page for everyone else: what it is, what it decided and why,
and a box to try it in. It calls the same public `/rag-ask` endpoint the API
docs describe, so what a visitor sees is the real service, not a mock.
"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "static" / "home.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html")
