from sqlalchemy.orm import Session

from app.core.metrics import FREE_ANSWER_EVALUATIONS_TOTAL
from app.models.entities import FreeEvaluation
from app.repositories.question_repository import QuestionRepository
from app.repositories.session_repository import SessionRepository
from app.schemas.evaluation import EvaluationResult
from app.services.openai_service import OpenAIService


class EvaluationService:
    def __init__(self, db: Session, openai_service: OpenAIService):
        self.db = db
        self.openai_service = openai_service
        self.questions = QuestionRepository(db)
        self.sessions = SessionRepository(db)

    async def evaluate_free_answer(self, session_id: str, question_id: str, answer: str) -> EvaluationResult:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("session not found")
        question = self.questions.get(question_id)
        if not question or question.session_id != session_id:
            raise ValueError("question not found in this session")

        session.mode = "free"
        result = await self.openai_service.evaluate_answer(
            question=question.clean_text,
            reference_answer=question.explanation,
            user_answer=answer,
        )
        self.sessions.add_free_evaluation(
            FreeEvaluation(
                session_id=session_id,
                question_id=question_id,
                user_answer=answer,
                score=result.score,
                evaluation=result.model_dump(),
            )
        )
        self.db.commit()
        FREE_ANSWER_EVALUATIONS_TOTAL.inc()
        return result
