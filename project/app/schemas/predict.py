from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ResponseMeta


class PredictOperation(str, Enum):
    generate_tests = "generate_tests"
    evaluate_answer = "evaluate_answer"
    final_report = "final_report"


class PredictRequest(BaseModel):
    operation: PredictOperation
    payload: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    operation: PredictOperation
    data: dict[str, Any]
    meta: ResponseMeta
