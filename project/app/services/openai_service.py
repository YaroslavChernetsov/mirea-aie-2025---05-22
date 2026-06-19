import json
import logging
import time
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.metrics import AI_ERRORS_TOTAL, AI_REQUEST_DURATION_SECONDS, AI_REQUESTS_TOTAL
from app.schemas.evaluation import EvaluationResult
from app.schemas.question import QuestionImportAIResponse


T = TypeVar("T", bound=BaseModel)


class AIProviderUnavailable(Exception):
    pass


class AIInvalidJSON(Exception):
    pass


class OpenAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_configured else None
        self.logger = logging.getLogger("interviewtrainer.openai")

    async def generate_questions(self, raw_text: str, max_questions: int) -> QuestionImportAIResponse:
        prompt = f"""
You are an expert technical interview trainer.

Input contains interview questions in free form. Split it into separate questions, clean duplicates/noise,
preserve code formatting exactly using triple backticks when code exists, and generate a multiple-choice
test version for each question.

For every question return:
- raw_text: original question fragment
- clean_text: cleaned question text, with code blocks preserved
- question_type: one of theory, coding, output_prediction, system_design, sql, other
- options: exactly 4 meaningful answer options, not random
- correct_index: integer 0..3
- explanation: concise explanation of the correct answer
- topic: short technical topic
- difficulty: easy, medium, or hard

Limit to {max_questions} questions. Return strict JSON without markdown:
{{"questions":[...]}}

Input:
{raw_text}
"""
        return await self._json_completion("generate_tests", prompt, QuestionImportAIResponse)

    async def evaluate_answer(self, question: str, reference_answer: str, user_answer: str) -> EvaluationResult:
        prompt = f"""
You are evaluating a free-form answer for a technical interview question.
Consider both the final answer and the user's explanation/reasoning.

Question:
{question}

Reference answer/explanation:
{reference_answer}

User answer:
{user_answer}

Return strict JSON without markdown:
{{
  "score": 0-100,
  "correct_parts": ["..."],
  "wrong_parts": ["..."],
  "missing_parts": ["..."],
  "improvement_advice": ["..."],
  "ideal_answer": "...",
  "verdict": "..."
}}
"""
        return await self._json_completion("evaluate_answer", prompt, EvaluationResult)

    async def final_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
Create a concise preparation report from these evaluated interview answers.
Return strict JSON without markdown with keys:
average_score, strong_topics, weak_topics, recommendations.

Data:
{json.dumps(payload, ensure_ascii=False)}
"""
        return await self._json_dict("final_report", prompt)

    async def _json_dict(self, operation: str, prompt: str) -> dict[str, Any]:
        data = await self._raw_json(operation, prompt)
        if not isinstance(data, dict):
            raise AIInvalidJSON("AI response must be a JSON object")
        return data

    async def _json_completion(self, operation: str, prompt: str, schema: type[T]) -> T:
        data = await self._raw_json(operation, prompt)
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            AI_ERRORS_TOTAL.labels(operation, "schema_validation").inc()
            raise AIInvalidJSON(f"AI response did not match expected schema: {exc}") from exc

    async def _raw_json(self, operation: str, prompt: str) -> Any:
        first_error: Exception | None = None
        for attempt in range(2):
            retry_suffix = ""
            if attempt == 1:
                retry_suffix = (
                    "\n\nYour previous response was not valid JSON or did not match the schema. "
                    "Return only one valid JSON object. No markdown. No comments. No trailing commas."
                )
            try:
                content = await self._call_openai(operation, prompt + retry_suffix)
                return json.loads(content)
            except json.JSONDecodeError as exc:
                first_error = exc
                AI_ERRORS_TOTAL.labels(operation, "invalid_json").inc()
                continue
        raise AIInvalidJSON("OpenAI returned invalid JSON twice") from first_error

    async def _call_openai(self, operation: str, prompt: str) -> str:
        if not self.client:
            AI_ERRORS_TOTAL.labels(operation, "not_configured").inc()
            raise AIProviderUnavailable("OPENAI_API_KEY is not configured")

        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You return reliable, strict JSON for an interview preparation service.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=self.settings.request_timeout_seconds,
            )
            content = response.choices[0].message.content
            if not content:
                raise AIInvalidJSON("OpenAI returned empty content")
            AI_REQUESTS_TOTAL.labels(operation, "success").inc()
            return content
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            AI_REQUESTS_TOTAL.labels(operation, "error").inc()
            AI_ERRORS_TOTAL.labels(operation, exc.__class__.__name__).inc()
            raise AIProviderUnavailable("OpenAI is temporarily unavailable") from exc
        except APIStatusError as exc:
            AI_REQUESTS_TOTAL.labels(operation, "error").inc()
            AI_ERRORS_TOTAL.labels(operation, f"status_{exc.status_code}").inc()
            if 500 <= exc.status_code < 600 or exc.status_code == 429:
                raise AIProviderUnavailable("OpenAI is temporarily unavailable") from exc
            raise
        finally:
            AI_REQUEST_DURATION_SECONDS.labels(operation).observe(time.perf_counter() - started)
