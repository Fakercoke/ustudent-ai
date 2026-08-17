"""Lesson 3 · POST /can-graduate

Decide whether a student can graduate given their credits and GPA.

Business rule:
    credits >= 120  AND  gpa >= 2.0  ->  can_graduate = True
    otherwise                        ->  can_graduate = False

Fill in the TODOs below. Then wire this router into app/main.py:

    from app.routes import can_graduate
    app.include_router(can_graduate.router, tags=["lesson-3"])
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# TODO 1: the request — the "form" the customer hands in.
class GraduationRequest(BaseModel):
    credits: float
    gpa: float

    def is_eligible(self) -> bool:
        """A student can graduate iff credits >= 120 AND gpa >= 2.0."""
        return self.credits >= 120 and self.gpa >= 2.0


# TODO 2: the response — the "answer" handed back.
class GraduationResponse(BaseModel):
    can_graduate: bool


# TODO 3: the endpoint — the service window itself.
@router.post("/can-graduate", response_model=GraduationResponse)
def check(req: GraduationRequest) -> GraduationResponse:
    return GraduationResponse(can_graduate=req.is_eligible())
