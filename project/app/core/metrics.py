import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)
AI_REQUESTS_TOTAL = Counter(
    "ai_requests_total",
    "Total AI provider requests.",
    ["operation", "status"],
)
AI_REQUEST_DURATION_SECONDS = Histogram(
    "ai_request_duration_seconds",
    "AI provider request duration in seconds.",
    ["operation"],
)
AI_ERRORS_TOTAL = Counter(
    "ai_errors_total",
    "Total AI provider errors.",
    ["operation", "error_type"],
)
TEST_SESSIONS_TOTAL = Counter("test_sessions_total", "Total started test sessions.")
FREE_ANSWER_EVALUATIONS_TOTAL = Counter(
    "free_answer_evaluations_total",
    "Total free answer evaluations.",
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            path_template = getattr(request.scope.get("route"), "path", request.url.path)
            HTTP_REQUESTS_TOTAL.labels(request.method, path_template, str(status_code)).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path_template).observe(
                time.perf_counter() - started
            )


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
