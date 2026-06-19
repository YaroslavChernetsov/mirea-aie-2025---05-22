from pydantic import BaseModel, Field

from app.schemas.common import ResponseMeta
from app.schemas.evaluation import EvaluationResult
from app.schemas.question import Difficulty, QuestionType


class PublicQuestion(BaseModel):
    id: str
    clean_text: str
    question_type: QuestionType
    options: list[str] = Field(min_length=4, max_length=4)
    topic: str
    difficulty: Difficulty


class StartTestRequest(BaseModel):
    source_session_id: str
    question_count: int | None = Field(default=None, ge=1)


class StartTestResponse(BaseModel):
    session_id: str
    total_questions: int
    current_question: PublicQuestion


class TestAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    selected_index: int = Field(ge=0, le=3)


class TestAnswerResponse(BaseModel):
    session_id: str
    accepted: bool
    completed: bool
    answered_count: int
    total_questions: int
    next_question: PublicQuestion | None = None


class TestMistake(BaseModel):
    question_id: str
    question: str
    selected_answer: str
    correct_answer: str
    explanation: str
    topic: str


class TestResults(BaseModel):
    session_id: str
    mode: str
    total_questions: int
    answered_count: int
    correct_count: int
    percent: float
    mistakes: list[TestMistake]


class FreeResults(BaseModel):
    session_id: str
    mode: str
    answered_count: int
    average_score: float
    strong_topics: list[str]
    weak_topics: list[str]
    recommendations: list[str]
    evaluations: list[EvaluationResult]


class ResultsResponse(BaseModel):
    result: TestResults | FreeResults
    meta: ResponseMeta | None = None
