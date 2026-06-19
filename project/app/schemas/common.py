from pydantic import BaseModel, Field


class ResponseMeta(BaseModel):
    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class ErrorResponse(BaseModel):
    detail: str
    correlation_id: str
