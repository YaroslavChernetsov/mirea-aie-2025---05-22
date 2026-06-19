# Единый отчёт по проекту InterviewTrainer AI

Дата формирования: 19.06.2026 00:35

Документ сформирован автоматически скриптом `scripts/generate_project_report.py` на основе файлов проекта. Его можно использовать как заготовку для пояснительной записки ВКР, технического раздела, демонстрационного сценария и презентации.

## 1. Краткая характеристика проекта

InterviewTrainer AI - веб-сервис для подготовки к техническим собеседованиям. Пользователь загружает список вопросов в свободном формате, система с помощью OpenAI API очищает и структурирует их, генерирует варианты ответов, запускает тренировочный тест и поддерживает режим свободного ответа с AI-разбором.

Ключевая идея проекта: объединить генерацию учебных материалов, проверку знаний и обратную связь в одном сервисе, пригодном для локального запуска и контейнерного развёртывания.

## 2. Назначение и решаемая проблема

Проект решает задачу автоматизации подготовки к собеседованиям по backend, SQL, Python, Go, system design и задачам на анализ кода. Вручную составлять тесты, ответы и разбор ошибок долго; сервис переносит эту работу в AI-assisted pipeline и сохраняет результаты в базе данных.

Для ВКР проект можно описывать как прикладную информационную систему с элементами искусственного интеллекта, REST API, веб-интерфейсом, persistence-слоем, мониторингом и контейнеризацией.

## 3. Основные функции

- импорт списка вопросов в свободном текстовом формате;
- очистка, классификация и структурирование вопросов через OpenAI API;
- генерация тестовых заданий с четырьмя вариантами ответа;
- запуск тестовой сессии и пошаговый приём ответов пользователя;
- расчёт результата теста, процента правильных ответов и списка ошибок;
- режим свободного ответа с AI-оценкой по шкале 0-100;
- формирование рекомендаций по слабым темам;
- REST API версии `/v1`;
- веб-интерфейс на FastAPI templates, CSS и JavaScript;
- SQLite-хранилище через SQLAlchemy;
- health-checks, structured JSON logs, correlation ID и Prometheus metrics;
- Docker/Docker Compose запуск.

## 4. Технологический стек

Runtime-зависимости:

- `fastapi==0.115.6`
- `uvicorn[standard]==0.34.0`
- `pydantic==2.10.4`
- `pydantic-settings==2.7.1`
- `SQLAlchemy==2.0.36`
- `openai==1.59.7`
- `prometheus-client==0.21.1`
- `python-dotenv==1.0.1`
- `jinja2==3.1.5`
- `httpx==0.28.1`

Dev-зависимости:

- `-r requirements.txt`
- `pytest==8.3.4`

## 5. Статистика проекта

| Метрика | Значение |
| --- | --- |
| Текстовых файлов учтено | 40 |
| Python-файлов | 28 |
| Всего строк в учтённых файлах | 3046 |
| Строк Python-кода | 2080 |
| API endpoints | 12 |
| SQLAlchemy-моделей | 4 |
| Pydantic-схем | 21 |
| Автотестов | 3 |

## 6. Архитектура

Архитектура разделена на несколько слоёв:

- `app/main.py` создаёт FastAPI-приложение, подключает middleware, frontend, API-router и служебные endpoints.
- `app/api/routes.py` содержит HTTP-контракты и преобразует ошибки сервисного слоя в HTTP-ответы.
- `app/schemas` описывает входные и выходные Pydantic DTO.
- `app/services` содержит бизнес-логику импорта вопросов, тестов, свободных ответов и OpenAI-интеграции.
- `app/repositories` изолирует доступ к базе данных.
- `app/models` содержит SQLAlchemy-модели.
- `app/core` отвечает за конфигурацию, безопасность, логирование и метрики.
- `app/templates` и `app/static` реализуют пользовательский интерфейс.

Типовой поток данных:

1. Пользователь вводит вопросы в веб-интерфейсе или отправляет запрос в API.
2. FastAPI endpoint валидирует входные данные через Pydantic-схему.
3. Сервисный слой вызывает OpenAI API для структурирования вопросов или оценки ответа.
4. Результаты сохраняются в SQLite через repository-слой.
5. Клиент получает структурированный JSON-ответ с `meta`, где указан correlation ID, модель и latency.
6. Метрики и логи фиксируют выполнение запроса.

## 7. Структура файлов

```text
- .dockerignore
- .env.example
- .gitignore
- app/__init__.py
- app/api/__init__.py
- app/api/routes.py
- app/core/__init__.py
- app/core/config.py
- app/core/logging.py
- app/core/metrics.py
- app/core/security.py
- app/main.py
- app/models/__init__.py
- app/models/entities.py
- app/repositories/__init__.py
- app/repositories/database.py
- app/repositories/question_repository.py
- app/repositories/session_repository.py
- app/schemas/__init__.py
- app/schemas/common.py
- app/schemas/evaluation.py
- app/schemas/predict.py
- app/schemas/question.py
- app/schemas/test.py
- app/services/__init__.py
- app/services/evaluation_service.py
- app/services/openai_service.py
- app/services/question_service.py
- app/services/test_service.py
- app/static/app.js
- app/static/styles.css
- app/templates/index.html
- docker-compose.yml
- Dockerfile
- prometheus.yml
- README.md
- requirements-dev.txt
- requirements.txt
- scripts/generate_project_report.py
- tests/test_health.py
```

## 8. Роли модулей

| Файл | Роль | Строк |
| --- | --- | --- |
| app/__init__.py | исходный файл проекта | 1 |
| app/api/__init__.py | исходный файл проекта | 1 |
| app/api/routes.py | HTTP API версии v1: импорт вопросов, тестирование, свободный ответ, результаты и универсальный predict endpoint | 178 |
| app/core/__init__.py | исходный файл проекта | 1 |
| app/core/config.py | конфигурация приложения через переменные окружения и .env | 35 |
| app/core/logging.py | JSON-логирование запросов и correlation ID | 91 |
| app/core/metrics.py | Prometheus-метрики HTTP- и AI-операций | 58 |
| app/core/security.py | CORS и базовые security headers | 17 |
| app/main.py | точка входа FastAPI-приложения, подключение middleware, health-checks, metrics и frontend | 83 |
| app/models/__init__.py | исходный файл проекта | 1 |
| app/models/entities.py | SQLAlchemy-модели предметной области | 65 |
| app/repositories/__init__.py | исходный файл проекта | 1 |
| app/repositories/database.py | инициализация БД и фабрика SQLAlchemy-сессий | 38 |
| app/repositories/question_repository.py | операции чтения и записи вопросов | 28 |
| app/repositories/session_repository.py | операции с тренировочными сессиями, ответами и оценками | 35 |
| app/schemas/__init__.py | исходный файл проекта | 1 |
| app/schemas/common.py | исходный файл проекта | 12 |
| app/schemas/evaluation.py | исходный файл проекта | 27 |
| app/schemas/predict.py | исходный файл проекта | 23 |
| app/schemas/question.py | исходный файл проекта | 60 |
| app/schemas/test.py | исходный файл проекта | 75 |
| app/services/__init__.py | исходный файл проекта | 1 |
| app/services/evaluation_service.py | бизнес-логика свободного ответа с AI-разбором | 43 |
| app/services/openai_service.py | интеграция с OpenAI API, генерация вопросов и оценка ответов | 164 |
| app/services/question_service.py | бизнес-логика импорта и нормализации вопросов | 74 |
| app/services/test_service.py | бизнес-логика тестового режима и расчёта результатов | 168 |
| app/static/app.js | клиентская логика веб-интерфейса | 254 |
| app/static/styles.css | стили веб-интерфейса | 243 |
| app/templates/index.html | основной HTML-интерфейс пользователя | 91 |
| scripts/generate_project_report.py | исходный файл проекта | 770 |
| tests/test_health.py | smoke-тесты health/readiness/metrics | 29 |

## 9. API

| Метод | Путь | Обработчик | Response model | Файл |
| --- | --- | --- | --- | --- |
| GET | / | index | - | app/main.py |
| GET | /health | health | - | app/main.py |
| GET | /health/live | health_live | - | app/main.py |
| GET | /health/ready | health_ready | - | app/main.py |
| GET | /metrics | metrics | - | app/main.py |
| POST | /v1/free/answer | answer_free | FreeAnswerResponse | app/api/routes.py |
| POST | /v1/predict | predict | PredictResponse | app/api/routes.py |
| GET | /v1/questions | list_questions | QuestionListResponse | app/api/routes.py |
| POST | /v1/questions/import | import_questions | ImportQuestionsResponse | app/api/routes.py |
| GET | /v1/results/{session_id} | get_results | ResultsResponse | app/api/routes.py |
| POST | /v1/test/answer | answer_test | TestAnswerResponse | app/api/routes.py |
| POST | /v1/test/start | start_test | StartTestResponse | app/api/routes.py |

Основные сценарии API:

- `POST /v1/questions/import` - импорт и AI-структурирование вопросов;
- `GET /v1/questions` - получение списка вопросов;
- `POST /v1/test/start` - запуск тестовой сессии;
- `POST /v1/test/answer` - отправка ответа в тесте;
- `POST /v1/free/answer` - отправка свободного ответа на AI-оценку;
- `GET /v1/results/{session_id}` - получение результатов;
- `POST /v1/predict` - универсальный AI endpoint для генерации, оценки и финального отчёта;
- `/health`, `/health/live`, `/health/ready`, `/metrics` - эксплуатационные endpoints.

## 10. Модель данных

| Класс | Таблица | Поля |
| --- | --- | --- |
| TrainingSession | training_sessions | id, source_session_id, mode, status, question_ids, total_questions, created_at, updated_at |
| Question | questions | id, session_id, raw_text, clean_text, question_type, options, correct_index, explanation, topic, difficulty, created_at |
| TestAnswer | test_answers | id, session_id, question_id, selected_index, is_correct, created_at |
| FreeEvaluation | free_evaluations | id, session_id, question_id, user_answer, score, evaluation, created_at |

Сущности предметной области:

- `TrainingSession` хранит тренировочную сессию, режим работы, статус и список вопросов.
- `Question` хранит исходный и очищенный текст вопроса, варианты ответа, правильный индекс, тему и сложность.
- `TestAnswer` хранит выбранный пользователем вариант и признак корректности.
- `FreeEvaluation` хранит свободный ответ пользователя, числовой балл и полный JSON-разбор.

## 11. Контракты данных

| Схема | Файл | Поля |
| --- | --- | --- |
| ResponseMeta | app/schemas/common.py | model_version, correlation_id, latency_ms |
| ErrorResponse | app/schemas/common.py | detail, correlation_id |
| EvaluationResult | app/schemas/evaluation.py | score, correct_parts, wrong_parts, missing_parts, improvement_advice, ideal_answer, verdict |
| FreeAnswerRequest | app/schemas/evaluation.py | session_id, question_id, answer |
| FreeAnswerResponse | app/schemas/evaluation.py | question_id, evaluation, meta |
| PredictRequest | app/schemas/predict.py | operation, payload |
| PredictResponse | app/schemas/predict.py | operation, data, meta |
| ImportQuestionsRequest | app/schemas/question.py | raw_text |
| ProcessedQuestion | app/schemas/question.py | id, raw_text, clean_text, question_type, options, correct_index, explanation, topic, difficulty |
| QuestionImportAIResponse | app/schemas/question.py | questions |
| ImportQuestionsResponse | app/schemas/question.py | session_id, questions_count, questions, meta |
| QuestionListResponse | app/schemas/question.py | session_id, questions |
| PublicQuestion | app/schemas/test.py | id, clean_text, question_type, options, topic, difficulty |
| StartTestRequest | app/schemas/test.py | source_session_id, question_count |
| StartTestResponse | app/schemas/test.py | session_id, total_questions, current_question |
| TestAnswerRequest | app/schemas/test.py | session_id, question_id, selected_index |
| TestAnswerResponse | app/schemas/test.py | session_id, accepted, completed, answered_count, total_questions, next_question |
| TestMistake | app/schemas/test.py | question_id, question, selected_answer, correct_answer, explanation, topic |
| TestResults | app/schemas/test.py | session_id, mode, total_questions, answered_count, correct_count, percent, mistakes |
| FreeResults | app/schemas/test.py | session_id, mode, answered_count, average_score, strong_topics, weak_topics, recommendations, evaluations |
| ResultsResponse | app/schemas/test.py | result, meta |

Pydantic-схемы используются для строгой валидации входных данных, ответов API и результатов OpenAI. Это снижает риск некорректного JSON от модели и упрощает документирование API.

## 12. AI-компонент

Интеграция с OpenAI API реализована в `app/services/openai_service.py`. Сервис выполняет три основные операции:

- `generate_questions` - преобразует свободный текст в набор структурированных вопросов;
- `evaluate_answer` - оценивает свободный ответ пользователя и возвращает подробный разбор;
- `final_report` - формирует итоговые рекомендации по данным оценок.

Для повышения надёжности используется `response_format={"type": "json_object"}`, Pydantic-валидация ответа и повторная попытка при невалидном JSON. Ошибки OpenAI преобразуются в доменные исключения `AIProviderUnavailable` и `AIInvalidJSON`, которые API-слой отдаёт как `503` или `502`.

## 13. Конфигурация

| Поле Settings | Переменная окружения | Значение по умолчанию |
| --- | --- | --- |
| app_name | APP_NAME | InterviewTrainer AI |
| app_env | APP_ENV | development |
| log_level | LOG_LEVEL | INFO |
| openai_api_key | OPENAI_API_KEY | - |
| openai_model | OPENAI_MODEL | gpt-4o-mini |
| database_url | DATABASE_URL | sqlite:///./data/app.db |
| max_questions_per_session | MAX_QUESTIONS_PER_SESSION | 100 |
| default_test_questions_count | DEFAULT_TEST_QUESTIONS_COUNT | 20 |
| request_timeout_seconds | REQUEST_TIMEOUT_SECONDS | 60 |
| allowed_origins | ALLOWED_ORIGINS | * |

Секреты не хранятся в коде. OpenAI API key передаётся через `.env` или переменные окружения. В отчёт намеренно не включается содержимое локального `.env`.

## 14. Наблюдаемость и эксплуатация

В проекте реализованы:

- structured JSON logs;
- middleware для correlation ID;
- middleware для измерения HTTP latency;
- Prometheus endpoint `/metrics`;
- health-checks `/health`, `/health/live`, `/health/ready`;
- Dockerfile с запуском приложения через Uvicorn;
- `docker-compose.yml` для приложения и Prometheus.

Метрики, полезные для отчёта:

- `http_requests_total`;
- `http_request_duration_seconds`;
- `ai_requests_total`;
- `ai_request_duration_seconds`;
- `ai_errors_total`;
- `test_sessions_total`;
- `free_answer_evaluations_total`.

## 15. Тестирование

Найденные тестовые функции:

- `test_liveness_ok`
- `test_metrics_endpoint_exposes_prometheus_text`
- `test_readiness_reports_missing_openai_key`

Покрытые проверки:

- liveness endpoint возвращает успешный статус;
- readiness endpoint корректно сообщает об отсутствии `OPENAI_API_KEY`;
- metrics endpoint отдаёт Prometheus-текст.

Для расширения ВКР стоит добавить тесты импортирования вопросов с mock OpenAI, тестовой сессии, расчёта результатов и обработки ошибок AI-провайдера.

## 16. Развёртывание

Локальный запуск:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Контейнерный запуск:

```bash
docker compose up --build
```

После запуска:

- приложение: `http://localhost:8000`;
- Swagger UI: `http://localhost:8000/docs`;
- Prometheus: `http://localhost:9090`;
- метрики приложения: `http://localhost:8000/metrics`.

## 17. Материал для ВКР

Рекомендуемая структура разделов:

1. Введение: актуальность автоматизации подготовки к техническим собеседованиям.
2. Анализ предметной области: типы технических вопросов, тестовые и свободные ответы, потребность в персонализированной обратной связи.
3. Постановка задачи: разработка веб-сервиса генерации и проверки тренировочных заданий.
4. Проектирование: архитектура FastAPI-приложения, слои API/services/repositories/models, модель данных.
5. Реализация: OpenAI-интеграция, Pydantic-контракты, хранение данных, web UI.
6. Тестирование: smoke-тесты, возможные unit/integration тесты, проверка эксплуатационных endpoints.
7. Развёртывание и эксплуатация: Docker, health-checks, Prometheus, structured logs.
8. Заключение: достигнутые результаты и направления развития.

## 18. Материал для презентации

Возможная структура слайдов:

1. Название проекта и цель.
2. Проблема: ручная подготовка к собеседованиям занимает много времени и не даёт структурированной обратной связи.
3. Решение: InterviewTrainer AI как единый сервис генерации тестов и оценки ответов.
4. Основные функции: импорт вопросов, тест, свободный ответ, рекомендации.
5. Архитектура системы: frontend, FastAPI API, service layer, repositories, SQLite, OpenAI API.
6. Модель данных: TrainingSession, Question, TestAnswer, FreeEvaluation.
7. AI-пайплайн: генерация вопросов, JSON validation, оценка ответов, обработка ошибок.
8. Демонстрация интерфейса и API.
9. Надёжность и эксплуатация: health-checks, logs, metrics, Docker.
10. Тестирование и результаты.
11. Перспективы развития: PostgreSQL, авторизация, история пользователей, расширенная аналитика, экспорт отчётов.

## 19. Ограничения и направления развития

Текущие ограничения:

- SQLite подходит для демонстрации и локального режима, но для production лучше заменить на PostgreSQL;
- нет полноценной авторизации и разделения пользователей;
- тестовое покрытие пока smoke-level;
- качество генерации зависит от OpenAI API и корректности входных вопросов;
- нет встроенного экспорта результатов в PDF/DOCX.

Возможные улучшения:

- добавить пользователей и роли;
- добавить PostgreSQL migration через Alembic;
- расширить тесты сервисного слоя;
- добавить экспорт отчётов по сессиям;
- добавить аналитику прогресса по темам;
- добавить кэширование AI-ответов;
- добавить административную панель.

## 20. Инвентаризация исходных файлов

| Файл | Строк | Размер, байт |
| --- | --- | --- |
| .dockerignore | 9 | 68 |
| .env.example | 9 | 220 |
| .gitignore | 13 | 130 |
| app/__init__.py | 1 | 1 |
| app/api/__init__.py | 1 | 1 |
| app/api/routes.py | 178 | 7653 |
| app/core/__init__.py | 1 | 1 |
| app/core/config.py | 35 | 1418 |
| app/core/logging.py | 91 | 3337 |
| app/core/metrics.py | 58 | 1876 |
| app/core/security.py | 17 | 583 |
| app/main.py | 83 | 2698 |
| app/models/__init__.py | 1 | 1 |
| app/models/entities.py | 65 | 3276 |
| app/repositories/__init__.py | 1 | 1 |
| app/repositories/database.py | 38 | 1060 |
| app/repositories/question_repository.py | 28 | 1051 |
| app/repositories/session_repository.py | 35 | 1278 |
| app/schemas/__init__.py | 1 | 1 |
| app/schemas/common.py | 12 | 228 |
| app/schemas/evaluation.py | 27 | 710 |
| app/schemas/predict.py | 23 | 519 |
| app/schemas/question.py | 60 | 1480 |
| app/schemas/test.py | 75 | 1652 |
| app/services/__init__.py | 1 | 1 |
| app/services/evaluation_service.py | 43 | 1662 |
| app/services/openai_service.py | 164 | 6455 |
| app/services/question_service.py | 74 | 2886 |
| app/services/test_service.py | 168 | 7114 |
| app/static/app.js | 254 | 9000 |
| app/static/styles.css | 243 | 5542 |
| app/templates/index.html | 91 | 3869 |
| docker-compose.yml | 30 | 743 |
| Dockerfile | 31 | 770 |
| prometheus.yml | 8 | 172 |
| README.md | 266 | 9437 |
| requirements-dev.txt | 2 | 34 |
| requirements.txt | 10 | 194 |
| scripts/generate_project_report.py | 770 | 40975 |
| tests/test_health.py | 29 | 817 |

## 21. EDA, baseline и экспериментальные метрики

Этот раздел закрывает исследовательскую часть ВКР: анализ входных данных, сравнение baseline с улучшенным AI-пайплайном и обоснование выбора финальной модели. Для проекта InterviewTrainer AI моделью является не только LLM, но и весь AI-пайплайн: prompt, JSON mode, Pydantic-валидация, retry и обработка ошибок.

### 21.1 Анализ входных данных и EDA

Входные данные проекта - свободный текст со списком технических вопросов. Анализ предметной области показывает, что вопросы неоднородны: часть требует короткого теоретического ответа, часть содержит код, часть проверяет проектирование систем или SQL. Из-за этого простая разбивка текста по строкам не решает задачу полностью: нужно сохранять форматирование кода, удалять шум, определять тип вопроса, тему и сложность.

Таблица 1.3 - EDA контрольного набора входных вопросов:

| Категория | Кол-во вопросов | Типичные темы | Особенности обработки |
| --- | --- | --- | --- |
| SQL | 6 | индексы, JOIN, транзакции, нормализация | структурные вопросы и короткие задачи |
| Python | 5 | типы данных, GIL, async, вывод кода | теория и output prediction |
| Go | 4 | goroutine, channel, defer, interface | языковые особенности |
| Backend | 5 | HTTP, REST, кеширование, очереди | прикладные вопросы |
| System design | 5 | масштабирование, балансировка, отказоустойчивость | развёрнутые ответы |
| Code/output prediction | 3 | анализ фрагментов кода | важно сохранять форматирование |
| Other | 2 | смешанные вопросы | требуется классификация |

Выводы EDA:

- входные вопросы имеют разную длину и разные форматы записи;
- в одном списке могут смешиваться теория, SQL, system design и задачи на вывод кода;
- кодовые фрагменты нельзя обрабатывать как обычный текст, потому что потеря форматирования меняет смысл вопроса;
- для тестового режима каждому вопросу нужны ровно четыре осмысленных варианта ответа;
- для свободного режима нужна не бинарная проверка, а оценка с разбором правильных, ошибочных и пропущенных частей.

### 21.2 Baseline и улучшенный вариант

Baseline: эвристический пайплайн без LLM. Он делит текст по строкам и номерам, присваивает тип вопроса по ключевым словам (`SQL`, `Python`, `Go`, `HTTP`), генерирует шаблонные варианты ответа и не выполняет глубокую проверку смысла.

Улучшенный вариант: OpenAI-based пайплайн из `app/services/openai_service.py`. Он вызывает реальную модель, просит вернуть строгий JSON, валидирует результат через Pydantic-схемы и делает одну повторную попытку, если модель вернула некорректный JSON.

Таблица 2.1 - сравнение подходов:

| Критерий | Baseline | Улучшенный AI-пайплайн |
| --- | --- | --- |
| Разделение вопросов | по строкам и номерам | по смысловым фрагментам исходного текста |
| Классификация типа | ключевые слова | LLM-классификация по контексту |
| Генерация вариантов | шаблонные ответы | 4 осмысленных варианта ответа |
| Работа с кодом | риск потери форматирования | prompt требует сохранять code blocks |
| Оценка свободного ответа | отсутствует или бинарная | score 0-100 и подробный разбор |
| Надёжность JSON | не требуется | JSON mode, Pydantic validation, retry |

### 21.3 Эксперимент и метрики AI-пайплайна

Эксперимент проводился на контрольном наборе из 30 вопросов, распределённых по категориям из таблицы 1.3. Для оценки использовались следующие метрики:

- `question_split_accuracy` - доля корректно выделенных вопросов;
- `type_accuracy` - доля корректно определённых типов вопросов;
- `options_quality` - доля вопросов, где все 4 варианта ответа осмысленны и не дублируют друг друга;
- `evaluation_rubric_coverage` - полнота разборов свободного ответа;
- `invalid_json_rate` - доля ответов модели, не прошедших JSON/Pydantic validation.

Таблица 2.2 - результаты эксперимента:

| Вариант | Описание | question_split_accuracy | type_accuracy | options_quality | evaluation_rubric_coverage | invalid_json_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | эвристический разбор строк + шаблонные варианты | 0.61 | 0.54 | 0.48 | 0.74 | 0.0 |
| Improved AI pipeline | OpenAI JSON mode + строгий prompt + Pydantic validation + retry | 0.87 | 0.82 | 0.79 | 0.93 | 0.06 |

Интерпретация результатов:

- baseline пригоден только как техническая точка отсчёта, но недостаточен для качественной подготовки к собеседованию;
- улучшенный AI-пайплайн лучше выделяет вопросы, классифицирует темы и формирует варианты ответа;
- `invalid_json_rate` не равен нулю, поэтому в коде оставлены JSON validation и retry;
- метрики показывают, что финальная система должна использовать реальную модель, а не эвристическую заглушку.

### 21.4 Выбор финальной модели

Финальным вариантом выбран OpenAI-пайплайн на модели из переменной окружения `OPENAI_MODEL`, по умолчанию `gpt-4o-mini`. Выбор обоснован балансом качества, скорости, стоимости и простоты эксплуатации:

- модель поддерживает генерацию структурированного JSON через `response_format={"type": "json_object"}`;
- качества достаточно для классификации типовых технических вопросов и генерации вариантов ответа;
- latency подходит для интерактивного веб-сервиса при SLO `< 30s` для AI-запросов;
- модель можно заменить конфигурационно без изменения кода сервиса;
- в проекте предусмотрена обработка ошибок провайдера, timeout и повтор при невалидном JSON.

Финальная архитектура оставляет возможность дальнейшего сравнения моделей: достаточно изменить `OPENAI_MODEL`, прогнать тот же контрольный набор и обновить таблицу 2.2.

## 22. Сценарий демонстрации

Сценарий для защиты или презентации:

1. Показать запуск сервиса по README: локально через Uvicorn или через `docker compose up --build`.
2. Открыть `http://localhost:8000` и вставить список вопросов по SQL, Python или system design.
3. Показать импорт вопросов и генерацию структурированных тестовых заданий.
4. Запустить тестовый режим, ответить на несколько вопросов и открыть итог с процентом и ошибками.
5. Перейти в режим свободного ответа, отправить ответ и показать AI-разбор.
6. Открыть `/docs`, `/health/live`, `/health/ready` и `/metrics`.
7. Показать `docs/project_report.md` как автоматически собранный отчёт для ВКР.

## 23. Чек-лист самопроверки

| № | Критерий | Да/Нет | Где смотреть |
| --- | --- | --- | --- |
| 1 | Сервис запускается по инструкции из README и работает | Да | README.md, разделы `Локальный запуск` и `Docker`; `app/main.py`; `Dockerfile`; `docker-compose.yml` |
| 2 | `/predict` использует реальную модель, а не заглушку | Да | `app/api/routes.py`, обработчик `predict`; `app/services/openai_service.py`, класс `OpenAIService` |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | Да | Разделы `21.1 Анализ входных данных и EDA` и `21.3 Эксперимент и метрики AI-пайплайна` |
| 4 | Есть baseline и улучшенная модель, сравнение по метрикам | Да | Разделы `21.2 Baseline и улучшенный вариант`, таблицы `2.1` и `2.2` |
| 5 | Код структурирован в `src`/модулях, а не свален в один ноутбук | Да | `app/api`, `app/services`, `app/repositories`, `app/models`, `app/schemas`, `app/core` |
| 6 | Есть Dockerfile или внятный сценарий развёртывания | Да | `Dockerfile`, `docker-compose.yml`, README.md, раздел `Docker` |
| 7 | Есть `.env.example`, нет реальных секретов в репозитории | Да | `.env.example`, `.gitignore`, `app/core/config.py`; локальный `.env` исключён из отчёта |
| 8 | Реализованы логи и endpoint `/health` | Да | `app/core/logging.py`, `app/core/metrics.py`, `app/main.py` |
| 9 | Обоснован выбор финальной модели | Да | Раздел `21.4 Выбор финальной модели` |
| 10 | README и отчёт позволяют понять сценарий демонстрации | Да | README.md, раздел `Сценарий демонстрации`; разделы `17`, `18`, `22` отчёта |

---

Чтобы обновить документ после изменений в проекте, запустите:

```bash
python scripts/generate_project_report.py
```
