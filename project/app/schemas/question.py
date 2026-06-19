from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ResponseMeta


class QuestionType(str, Enum):
    theory = "theory"
    coding = "coding"
    output_prediction = "output_prediction"
    system_design = "system_design"
    sql = "sql"
    other = "other"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ImportQuestionsRequest(BaseModel):
    raw_text: Annotated[str, Field(min_length=5)]


class ProcessedQuestion(BaseModel):
    id: str | None = None
    raw_text: str = ""
    clean_text: str
    question_type: QuestionType
    options: Annotated[list[str], Field(min_length=4, max_length=4)]
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    topic: str
    difficulty: Difficulty

    @field_validator("options")
    @classmethod
    def non_empty_options(cls, options: list[str]) -> list[str]:
        if any(not option.strip() for option in options):
            raise ValueError("all options must be non-empty")
        return options


class QuestionImportAIResponse(BaseModel):
    questions: Annotated[list[ProcessedQuestion], Field(min_length=1)]


class ImportQuestionsResponse(BaseModel):
    session_id: str
    questions_count: int
    questions: list[ProcessedQuestion]
    meta: ResponseMeta


class QuestionListResponse(BaseModel):
    session_id: str | None = None
    questions: list[ProcessedQuestion]
