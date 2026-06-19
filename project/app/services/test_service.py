import random
from collections import defaultdict
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.metrics import TEST_SESSIONS_TOTAL
from app.models.entities import FreeEvaluation, TestAnswer, TrainingSession
from app.repositories.question_repository import QuestionRepository
from app.repositories.session_repository import SessionRepository
from app.schemas.evaluation import EvaluationResult
from app.schemas.question import Difficulty, QuestionType
from app.schemas.test import FreeResults, PublicQuestion, TestMistake, TestResults


__test__ = False


class TestService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.questions = QuestionRepository(db)
        self.sessions = SessionRepository(db)

    def start_test(self, source_session_id: str, question_count: int | None) -> tuple[str, PublicQuestion, int]:
        source_session = self.sessions.get(source_session_id)
        if not source_session:
            raise ValueError("source session not found")
        available_questions = self.questions.list_by_session(source_session_id)
        if not available_questions:
            raise ValueError("source session has no questions")

        requested = question_count or self.settings.default_test_questions_count
        total = min(requested, len(available_questions), self.settings.max_questions_per_session)
        selected = random.sample(available_questions, total)
        session_id = str(uuid4())
        session = TrainingSession(
            id=session_id,
            source_session_id=source_session_id,
            mode="test",
            status="active",
            question_ids=[question.id for question in selected],
            total_questions=total,
        )
        self.sessions.add(session)
        self.db.commit()
        TEST_SESSIONS_TOTAL.inc()
        return session_id, public_question(selected[0]), total

    def answer(self, session_id: str, question_id: str, selected_index: int) -> tuple[bool, int, int, PublicQuestion | None]:
        session = self.sessions.get(session_id)
        if not session or session.mode != "test":
            raise ValueError("test session not found")
        if question_id not in session.question_ids:
            raise ValueError("question does not belong to this test")
        if self.sessions.get_test_answer(session_id, question_id):
            raise RuntimeError("question already answered")

        question = self.questions.get(question_id)
        if not question:
            raise ValueError("question not found")

        answer = TestAnswer(
            session_id=session_id,
            question_id=question_id,
            selected_index=selected_index,
            is_correct=selected_index == question.correct_index,
        )
        self.sessions.add_test_answer(answer)
        self.db.commit()

        answers = self.sessions.get_test_answers(session_id)
        answered_ids = {item.question_id for item in answers}
        next_question = None
        for next_id in session.question_ids:
            if next_id not in answered_ids:
                next_entity = self.questions.get(next_id)
                next_question = public_question(next_entity) if next_entity else None
                break
        if next_question is None:
            session.status = "completed"
            self.db.commit()
        return next_question is None, len(answers), session.total_questions, next_question

    def test_results(self, session_id: str) -> TestResults:
        session = self.sessions.get(session_id)
        if not session or session.mode != "test":
            raise ValueError("test session not found")
        answers = self.sessions.get_test_answers(session_id)
        questions = {question.id: question for question in self.questions.list_by_ids(session.question_ids)}
        mistakes: list[TestMistake] = []
        correct_count = 0
        for answer in answers:
            question = questions.get(answer.question_id)
            if not question:
                continue
            if answer.is_correct:
                correct_count += 1
            else:
                mistakes.append(
                    TestMistake(
                        question_id=question.id,
                        question=question.clean_text,
                        selected_answer=question.options[answer.selected_index],
                        correct_answer=question.options[question.correct_index],
                        explanation=question.explanation,
                        topic=question.topic,
                    )
                )
        percent = round((correct_count / session.total_questions) * 100, 2) if session.total_questions else 0
        return TestResults(
            session_id=session_id,
            mode="test",
            total_questions=session.total_questions,
            answered_count=len(answers),
            correct_count=correct_count,
            percent=percent,
            mistakes=mistakes,
        )

    def free_results(self, session_id: str) -> FreeResults:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("session not found")
        evaluations = self.sessions.get_free_evaluations(session_id)
        questions = {question.id: question for question in self.questions.list_by_session(session_id)}
        scores_by_topic: dict[str, list[float]] = defaultdict(list)
        parsed: list[EvaluationResult] = []
        for item in evaluations:
            question = questions.get(item.question_id)
            if question:
                scores_by_topic[question.topic].append(item.score)
            parsed.append(EvaluationResult.model_validate(item.evaluation))

        average_score = round(sum(item.score for item in evaluations) / len(evaluations), 2) if evaluations else 0
        topic_averages = {
            topic: sum(scores) / len(scores)
            for topic, scores in scores_by_topic.items()
            if scores
        }
        strong_topics = [topic for topic, score in topic_averages.items() if score >= 75]
        weak_topics = [topic for topic, score in topic_averages.items() if score < 60]
        recommendations = [
            f"Review topic: {topic}" for topic in weak_topics
        ] or ["Keep practicing with clear reasoning, tradeoffs, and examples."]
        return FreeResults(
            session_id=session_id,
            mode="free",
            answered_count=len(evaluations),
            average_score=average_score,
            strong_topics=strong_topics,
            weak_topics=weak_topics,
            recommendations=recommendations,
            evaluations=parsed,
        )


def public_question(question) -> PublicQuestion:
    return PublicQuestion(
        id=question.id,
        clean_text=question.clean_text,
        question_type=QuestionType(question.question_type),
        options=question.options,
        topic=question.topic,
        difficulty=Difficulty(question.difficulty),
    )
