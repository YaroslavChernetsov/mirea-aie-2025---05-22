from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.common import ResponseMeta


class EvaluationResult(BaseModel):
    score: float = Field(ge=0, le=100)
    correct_parts: list[str] = Field(default_factory=list)
    wrong_parts: list[str] = Field(default_factory=list)
    missing_parts: list[str] = Field(default_factory=list)
    improvement_advice: list[str] = Field(default_factory=list)
    ideal_answer: str
    verdict: str


class FreeAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: Annotated[str, Field(min_length=1)]


class FreeAnswerResponse(BaseModel):
    question_id: str
    evaluation: EvaluationResult
    meta: ResponseMeta
