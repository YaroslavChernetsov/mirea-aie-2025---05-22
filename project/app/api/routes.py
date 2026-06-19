import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_correlation_id
from app.repositories.database import get_db
from app.schemas.common import ResponseMeta
from app.schemas.evaluation import FreeAnswerRequest, FreeAnswerResponse
from app.schemas.predict import PredictOperation, PredictRequest, PredictResponse
from app.schemas.question import ImportQuestionsRequest, ImportQuestionsResponse, QuestionListResponse
from app.schemas.test import ResultsResponse, StartTestRequest, StartTestResponse, TestAnswerRequest, TestAnswerResponse
from app.services.evaluation_service import EvaluationService
from app.services.openai_service import AIInvalidJSON, AIProviderUnavailable, OpenAIService
from app.services.question_service import QuestionService
from app.services.test_service import TestService


router = APIRouter(prefix="/v1", tags=["v1"])


def response_meta(settings: Settings, started: float) -> ResponseMeta:
    return ResponseMeta(
        model_version=settings.openai_model,
        correlation_id=get_correlation_id(),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def translate_ai_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AIProviderUnavailable):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if isinstance(exc, AIInvalidJSON):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI operation failed")


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictResponse:
    started = time.perf_counter()
    openai_service = OpenAIService(settings)
    try:
        if request.operation == PredictOperation.generate_tests:
            raw_text = str(request.payload.get("raw_text") or request.payload.get("text") or "")
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="payload.raw_text is required")
            data = (await openai_service.generate_questions(raw_text, settings.max_questions_per_session)).model_dump()
        elif request.operation == PredictOperation.evaluate_answer:
            data = (
                await openai_service.evaluate_answer(
                    question=str(request.payload.get("question") or ""),
                    reference_answer=str(request.payload.get("reference_answer") or request.payload.get("explanation") or ""),
                    user_answer=str(request.payload.get("user_answer") or request.payload.get("answer") or ""),
                )
            ).model_dump()
        else:
            data = await openai_service.final_report(request.payload)
    except HTTPException:
        raise
    except (AIProviderUnavailable, AIInvalidJSON) as exc:
        raise translate_ai_error(exc) from exc
    return PredictResponse(operation=request.operation, data=data, meta=response_meta(settings, started))


@router.post("/questions/import", response_model=ImportQuestionsResponse, status_code=status.HTTP_201_CREATED)
async def import_questions(
    request: ImportQuestionsRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ImportQuestionsResponse:
    started = time.perf_counter()
    service = QuestionService(db, settings, OpenAIService(settings))
    try:
        session_id, questions = await service.import_questions(request.raw_text)
    except (AIProviderUnavailable, AIInvalidJSON) as exc:
        raise translate_ai_error(exc) from exc
    return ImportQuestionsResponse(
        session_id=session_id,
        questions_count=len(questions),
        questions=questions,
        meta=response_meta(settings, started),
    )


@router.get("/questions", response_model=QuestionListResponse)
def list_questions(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    session_id: Annotated[str | None, Query()] = None,
) -> QuestionListResponse:
    service = QuestionService(db, settings, OpenAIService(settings))
    return QuestionListResponse(session_id=session_id, questions=service.list_questions(session_id))


@router.post("/test/start", response_model=StartTestResponse, status_code=status.HTTP_201_CREATED)
def start_test(
    request: StartTestRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StartTestResponse:
    service = TestService(db, settings)
    try:
        session_id, first_question, total = service.start_test(request.source_session_id, request.question_count)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StartTestResponse(session_id=session_id, total_questions=total, current_question=first_question)


@router.post("/test/answer", response_model=TestAnswerResponse)
def answer_test(
    request: TestAnswerRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TestAnswerResponse:
    service = TestService(db, settings)
    try:
        completed, answered_count, total, next_question = service.answer(
            request.session_id,
            request.question_id,
            request.selected_index,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TestAnswerResponse(
        session_id=request.session_id,
        accepted=True,
        completed=completed,
        answered_count=answered_count,
        total_questions=total,
        next_question=next_question,
    )


@router.post("/free/answer", response_model=FreeAnswerResponse)
async def answer_free(
    request: FreeAnswerRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FreeAnswerResponse:
    started = time.perf_counter()
    service = EvaluationService(db, OpenAIService(settings))
    try:
        evaluation = await service.evaluate_free_answer(request.session_id, request.question_id, request.answer)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (AIProviderUnavailable, AIInvalidJSON) as exc:
        raise translate_ai_error(exc) from exc
    return FreeAnswerResponse(
        question_id=request.question_id,
        evaluation=evaluation,
        meta=response_meta(settings, started),
    )


@router.get("/results/{session_id}", response_model=ResultsResponse)
def get_results(
    session_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResultsResponse:
    started = time.perf_counter()
    service = TestService(db, settings)
    try:
        session = service.sessions.get(session_id)
        if not session:
            raise ValueError("session not found")
        result = service.test_results(session_id) if session.mode == "test" else service.free_results(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResultsResponse(result=result, meta=response_meta(settings, started))
