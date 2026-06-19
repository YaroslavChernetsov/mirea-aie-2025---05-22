from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import Question, TrainingSession
from app.repositories.question_repository import QuestionRepository
from app.repositories.session_repository import SessionRepository
from app.schemas.question import ProcessedQuestion
from app.services.openai_service import OpenAIService


class QuestionService:
    def __init__(self, db: Session, settings: Settings, openai_service: OpenAIService):
        self.db = db
        self.settings = settings
        self.openai_service = openai_service
        self.questions = QuestionRepository(db)
        self.sessions = SessionRepository(db)

    async def import_questions(self, raw_text: str) -> tuple[str, list[ProcessedQuestion]]:
        ai_response = await self.openai_service.generate_questions(
            raw_text=raw_text,
            max_questions=self.settings.max_questions_per_session,
        )
        session_id = str(uuid4())
        session = TrainingSession(id=session_id, mode="import", status="active")
        self.sessions.add(session)

        entities: list[Question] = []
        processed: list[ProcessedQuestion] = []
        for item in ai_response.questions[: self.settings.max_questions_per_session]:
            question_id = str(uuid4())
            entity = Question(
                id=question_id,
                session_id=session_id,
                raw_text=item.raw_text or item.clean_text,
                clean_text=item.clean_text,
                question_type=item.question_type.value,
                options=item.options,
                correct_index=item.correct_index,
                explanation=item.explanation,
                topic=item.topic,
                difficulty=item.difficulty.value,
            )
            entities.append(entity)
            processed.append(item.model_copy(update={"id": question_id}))

        session.question_ids = [question.id for question in entities]
        session.total_questions = len(entities)
        self.questions.add_many(entities)
        self.db.commit()
        return session_id, processed

    def list_questions(self, session_id: str | None = None) -> list[ProcessedQuestion]:
        if session_id:
            entities = self.questions.list_by_session(session_id)
        else:
            entities = self.questions.list_all()
        return [question_to_schema(entity) for entity in entities]


def question_to_schema(entity: Question) -> ProcessedQuestion:
    return ProcessedQuestion(
        id=entity.id,
        raw_text=entity.raw_text,
        clean_text=entity.clean_text,
        question_type=entity.question_type,
        options=entity.options,
        correct_index=entity.correct_index,
        explanation=entity.explanation,
        topic=entity.topic,
        difficulty=entity.difficulty,
    )
