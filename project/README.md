# InterviewTrainer AI

Production-ready демонстрационный AI-сервис для подготовки к техническим собеседованиям. Пользователь вставляет список вопросов, сервис структурирует их через OpenAI API, генерирует тест с 4 вариантами ответа и поддерживает режим свободного ответа с AI-разбором.

## Функции

- импорт вопросов в свободном формате, включая SQL, backend, Go, Python, system design и code/output prediction;
- генерация очищенных вопросов, типов, тем, сложности, 4 осмысленных вариантов ответа и объяснений;
- тестовый режим с итогом, процентом, ошибками и правильными ответами;
- свободный режим с оценкой ответа, разбором ошибок, недостающих частей и идеальным ответом;
- FastAPI API с версионированием `/v1`;
- SQLite с репозиторным слоем, пригодным для замены на PostgreSQL;
- JSON-логи, correlation ID, Prometheus `/metrics`, health-checks;
- Dockerfile, docker-compose и Prometheus.

## Архитектура

```text
app/main.py                    FastAPI app, health, metrics, frontend
app/api/routes.py              /v1 API endpoints
app/core/                      config, logging, metrics, security
app/schemas/                   Pydantic request/response schemas
app/services/                  OpenAI, questions, test, evaluation logic
app/repositories/              DB access layer
app/models/                    SQLAlchemy models
app/static/, app/templates/    frontend
tests/                         smoke tests
```

## Требования

- Python 3.12+
- Docker и Docker Compose для контейнерного запуска
- OpenAI API key в переменной `OPENAI_API_KEY`

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
copy .env.example .env
```

Заполните `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Запуск:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000`.

## Docker

```bash
copy .env.example .env
# заполните OPENAI_API_KEY
docker compose up --build
```

Сервис: `http://localhost:8000`  
Prometheus: `http://localhost:9090`

## Отчёт для ВКР и самопроверка

Единый отчёт по проекту формируется автоматически:

```bash
python scripts/generate_project_report.py
```

Результат сохраняется в `docs/project_report.md`. В отчёт включены архитектура, API, модель данных, зависимости, конфигурация, EDA входных данных, baseline, улучшенный AI-пайплайн, экспериментальные метрики, обоснование выбора финальной модели, сценарий демонстрации и чек-лист самопроверки.

Ключевые разделы отчёта:

- `21.1 Анализ входных данных и EDA`;
- `21.2 Baseline и улучшенный вариант`;
- `21.3 Эксперимент и метрики AI-пайплайна`;
- `21.4 Выбор финальной модели`;
- `22. Сценарий демонстрации`;
- `23. Чек-лист самопроверки`.

## Сценарий демонстрации

1. Запустить сервис локально или через Docker.
2. Открыть `http://localhost:8000`.
3. Вставить список технических вопросов и выполнить импорт.
4. Показать сгенерированные варианты ответов и запустить тестовый режим.
5. Ответить на несколько вопросов и открыть итог с процентом и ошибками.
6. Проверить свободный ответ и показать AI-разбор.
7. Открыть `http://localhost:8000/docs`, `/health/live`, `/health/ready` и `/metrics`.
8. Показать `docs/project_report.md` как отчётную основу для ВКР и презентации.

## Переменные окружения

| Переменная | Назначение |
| --- | --- |
| `OPENAI_API_KEY` | ключ OpenAI API, обязателен для readiness и AI-операций |
| `OPENAI_MODEL` | модель, по умолчанию `gpt-4o-mini` |
| `APP_ENV` | `development` или `production` |
| `LOG_LEVEL` | уровень логирования |
| `DATABASE_URL` | SQLAlchemy URL, по умолчанию `sqlite:///./data/app.db` |
| `MAX_QUESTIONS_PER_SESSION` | лимит вопросов на импорт |
| `DEFAULT_TEST_QUESTIONS_COUNT` | количество вопросов в тесте по умолчанию |
| `REQUEST_TIMEOUT_SECONDS` | timeout OpenAI-запросов |
| `ALLOWED_ORIGINS` | CORS origins |

Секреты не хранятся в коде. `.env` добавлен в `.gitignore`.

## API

Все AI-ответы содержат:

```json
{
  "meta": {
    "model_version": "gpt-4o-mini",
    "correlation_id": "uuid-or-client-header",
    "latency_ms": 1234.5
  }
}
```

Если клиент передал `X-Correlation-ID`, сервис использует его в логах, ответах и HTTP-заголовке.

### Импорт вопросов

```bash
curl -X POST http://localhost:8000/v1/questions/import \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: demo-1" \
  -d "{\"raw_text\":\"1. Что такое индекс в SQL?\\n2. Что выведет код Python: print(1 + 2)\"}"
```

### Получить вопросы

```bash
curl "http://localhost:8000/v1/questions?session_id=<session_id>"
```

### Запустить тест

```bash
curl -X POST http://localhost:8000/v1/test/start \
  -H "Content-Type: application/json" \
  -d "{\"source_session_id\":\"<session_id>\",\"question_count\":10}"
```

### Ответить в тесте

```bash
curl -X POST http://localhost:8000/v1/test/answer \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"<test_session_id>\",\"question_id\":\"<question_id>\",\"selected_index\":0}"
```

### Свободный ответ

```bash
curl -X POST http://localhost:8000/v1/free/answer \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"<session_id>\",\"question_id\":\"<question_id>\",\"answer\":\"Индекс ускоряет поиск, но замедляет запись...\"}"
```

### Универсальный AI endpoint

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"operation\":\"generate_tests\",\"payload\":{\"raw_text\":\"Что такое goroutine?\"}}"
```

### Результаты

```bash
curl http://localhost:8000/v1/results/<session_id>
```

## Health и Metrics

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics
```

`/health/ready` возвращает `503`, если не задан `OPENAI_API_KEY` или база недоступна.

Prometheus-метрики:

- `http_requests_total`
- `http_request_duration_seconds`
- `ai_requests_total`
- `ai_request_duration_seconds`
- `ai_errors_total`
- `test_sessions_total`
- `free_answer_evaluations_total`

## Логирование

Логи пишутся в stdout в JSON. Поля:

- `timestamp`
- `level`
- `message`
- `correlation_id`
- `path`
- `method`
- `status_code`
- `latency_ms`
- `model_version`

`OPENAI_API_KEY` не логируется.

## SLI/SLO

SLI:

- p95 latency;
- error rate;
- availability.

SLO:

- p95 latency `< 2s` для обычных API-запросов без OpenAI;
- p95 latency `< 30s` для AI-запросов;
- error rate `< 1%`;
- availability `> 99%`.

Error budget: если error rate превышает SLO или latency деградирует за пределы SLO, релизы замораживаются до устранения причины и восстановления бюджета.

## Алерты

- AI error rate `> 5%` за 5 минут;
- p95 AI latency `> 30s` за 5 минут;
- readiness failed;
- общий error rate `> 1%`.

## Production-ready checklist

- [x] версионированный API `/v1`;
- [x] Pydantic-схемы для входа и выхода;
- [x] корректные HTTP-коды `4xx`, `5xx`, `503`, `502`;
- [x] OpenAI key только через env;
- [x] JSON validation и одна retry-попытка при невалидном JSON от модели;
- [x] structured JSON logs;
- [x] correlation ID middleware;
- [x] Prometheus metrics;
- [x] liveness/readiness probes;
- [x] Docker non-root runtime;
- [x] SQLite repository layer with PostgreSQL-ready boundary.

## Проверка

```bash
pytest
```
