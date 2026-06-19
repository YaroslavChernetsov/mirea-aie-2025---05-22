from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as v1_router
from app.core.config import get_settings
from app.core.logging import CorrelationIdMiddleware, RequestLoggingMiddleware, configure_logging, get_correlation_id
from app.core.metrics import MetricsMiddleware, metrics_response
from app.core.security import configure_security
from app.repositories.database import db_ready, init_db


settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="InterviewTrainer AI",
    version="1.0.0",
    description="Production-ready AI service for technical interview preparation.",
    lifespan=lifespan,
)

configure_security(app, settings)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware, model_version=settings.openai_model)
app.add_middleware(CorrelationIdMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
app.include_router(v1_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "correlation_id": get_correlation_id()},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_count": settings.default_test_questions_count},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    try:
        database_ok = db_ready()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"database is not ready: {exc}") from exc
    if not settings.openai_configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OPENAI_API_KEY is not configured")
    if not database_ok:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="database is not ready")
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()
