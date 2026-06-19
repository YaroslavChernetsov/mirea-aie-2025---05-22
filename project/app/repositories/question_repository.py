from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import Question


class QuestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_many(self, questions: list[Question]) -> None:
        self.db.add_all(questions)

    def list_by_session(self, session_id: str) -> list[Question]:
        return list(self.db.scalars(select(Question).where(Question.session_id == session_id)).all())

    def list_all(self) -> list[Question]:
        return list(self.db.scalars(select(Question)).all())

    def list_by_ids(self, question_ids: list[str]) -> list[Question]:
        if not question_ids:
            return []
        questions = list(self.db.scalars(select(Question).where(Question.id.in_(question_ids))).all())
        by_id = {question.id: question for question in questions}
        return [by_id[question_id] for question_id in question_ids if question_id in by_id]

    def get(self, question_id: str) -> Question | None:
        return self.db.get(Question, question_id)
