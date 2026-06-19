from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import FreeEvaluation, TestAnswer, TrainingSession


class SessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, session: TrainingSession) -> None:
        self.db.add(session)

    def get(self, session_id: str) -> TrainingSession | None:
        return self.db.get(TrainingSession, session_id)

    def add_test_answer(self, answer: TestAnswer) -> None:
        self.db.add(answer)

    def get_test_answers(self, session_id: str) -> list[TestAnswer]:
        return list(self.db.scalars(select(TestAnswer).where(TestAnswer.session_id == session_id)).all())

    def get_test_answer(self, session_id: str, question_id: str) -> TestAnswer | None:
        return self.db.scalar(
            select(TestAnswer).where(
                TestAnswer.session_id == session_id,
                TestAnswer.question_id == question_id,
            )
        )

    def add_free_evaluation(self, evaluation: FreeEvaluation) -> None:
        self.db.add(evaluation)

    def get_free_evaluations(self, session_id: str) -> list[FreeEvaluation]:
        return list(self.db.scalars(select(FreeEvaluation).where(FreeEvaluation.session_id == session_id)).all())
