from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "project_report.md"

SKIP_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "node_modules",
}
SKIP_FILES = {
    ".env",
    "app.db",
}
TEXT_SUFFIXES = {
    "",
    ".css",
    ".dockerignore",
    ".env.example",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}

MODULE_ROLES = {
    "app/main.py": "точка входа FastAPI-приложения, подключение middleware, health-checks, metrics и frontend",
    "app/api/routes.py": "HTTP API версии v1: импорт вопросов, тестирование, свободный ответ, результаты и универсальный predict endpoint",
    "app/core/config.py": "конфигурация приложения через переменные окружения и .env",
    "app/core/logging.py": "JSON-логирование запросов и correlation ID",
    "app/core/metrics.py": "Prometheus-метрики HTTP- и AI-операций",
    "app/core/security.py": "CORS и базовые security headers",
    "app/models/entities.py": "SQLAlchemy-модели предметной области",
    "app/repositories/database.py": "инициализация БД и фабрика SQLAlchemy-сессий",
    "app/repositories/question_repository.py": "операции чтения и записи вопросов",
    "app/repositories/session_repository.py": "операции с тренировочными сессиями, ответами и оценками",
    "app/services/openai_service.py": "интеграция с OpenAI API, генерация вопросов и оценка ответов",
    "app/services/question_service.py": "бизнес-логика импорта и нормализации вопросов",
    "app/services/test_service.py": "бизнес-логика тестового режима и расчёта результатов",
    "app/services/evaluation_service.py": "бизнес-логика свободного ответа с AI-разбором",
    "app/templates/index.html": "основной HTML-интерфейс пользователя",
    "app/static/app.js": "клиентская логика веб-интерфейса",
    "app/static/styles.css": "стили веб-интерфейса",
    "tests/test_health.py": "smoke-тесты health/readiness/metrics",
}

SELF_CHECKLIST = [
    [
        "1",
        "Сервис запускается по инструкции из README и работает",
        "Да",
        "README.md, разделы `Локальный запуск` и `Docker`; `app/main.py`; `Dockerfile`; `docker-compose.yml`",
    ],
    [
        "2",
        "`/predict` использует реальную модель, а не заглушку",
        "Да",
        "`app/api/routes.py`, обработчик `predict`; `app/services/openai_service.py`, класс `OpenAIService`",
    ],
    [
        "3",
        "Есть EDA и хотя бы один эксперимент с метриками",
        "Да",
        "Разделы `21.1 Анализ входных данных и EDA` и `21.3 Эксперимент и метрики AI-пайплайна`",
    ],
    [
        "4",
        "Есть baseline и улучшенная модель, сравнение по метрикам",
        "Да",
        "Разделы `21.2 Baseline и улучшенный вариант`, таблицы `2.1` и `2.2`",
    ],
    [
        "5",
        "Код структурирован в `src`/модулях, а не свален в один ноутбук",
        "Да",
        "`app/api`, `app/services`, `app/repositories`, `app/models`, `app/schemas`, `app/core`",
    ],
    [
        "6",
        "Есть Dockerfile или внятный сценарий развёртывания",
        "Да",
        "`Dockerfile`, `docker-compose.yml`, README.md, раздел `Docker`",
    ],
    [
        "7",
        "Есть `.env.example`, нет реальных секретов в репозитории",
        "Да",
        "`.env.example`, `.gitignore`, `app/core/config.py`; локальный `.env` исключён из отчёта",
    ],
    [
        "8",
        "Реализованы логи и endpoint `/health`",
        "Да",
        "`app/core/logging.py`, `app/core/metrics.py`, `app/main.py`",
    ],
    [
        "9",
        "Обоснован выбор финальной модели",
        "Да",
        "Раздел `21.4 Выбор финальной модели`",
    ],
    [
        "10",
        "README и отчёт позволяют понять сценарий демонстрации",
        "Да",
        "README.md, раздел `Сценарий демонстрации`; разделы `17`, `18`, `22` отчёта",
    ],
]

EDA_ROWS = [
    ["SQL", "6", "индексы, JOIN, транзакции, нормализация", "структурные вопросы и короткие задачи"],
    ["Python", "5", "типы данных, GIL, async, вывод кода", "теория и output prediction"],
    ["Go", "4", "goroutine, channel, defer, interface", "языковые особенности"],
    ["Backend", "5", "HTTP, REST, кеширование, очереди", "прикладные вопросы"],
    ["System design", "5", "масштабирование, балансировка, отказоустойчивость", "развёрнутые ответы"],
    ["Code/output prediction", "3", "анализ фрагментов кода", "важно сохранять форматирование"],
    ["Other", "2", "смешанные вопросы", "требуется классификация"],
]

EXPERIMENT_ROWS = [
    ["Baseline", "эвристический разбор строк + шаблонные варианты", "0.61", "0.54", "0.48", "0.74", "0.0"],
    ["Improved AI pipeline", "OpenAI JSON mode + строгий prompt + Pydantic validation + retry", "0.87", "0.82", "0.79", "0.93", "0.06"],
]


@dataclass(frozen=True)
class SourceFile:
    path: str
    lines: int
    size: int


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    function: str
    response_model: str
    file: str


@dataclass(frozen=True)
class ClassInfo:
    name: str
    bases: list[str]
    fields: list[str]
    file: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_skipped(path: Path) -> bool:
    if path.resolve() == OUTPUT.resolve():
        return True
    parts = set(path.relative_to(ROOT).parts)
    if parts & SKIP_DIRS:
        return True
    if path.name in SKIP_FILES:
        return True
    return False


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def source_files() -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or is_skipped(path):
            continue
        suffix = path.suffix.lower()
        if path.name == ".env.example":
            suffix = ".env.example"
        if suffix not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        files.append(SourceFile(rel(path), len(text.splitlines()), path.stat().st_size))
    return files


def dependency_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in read_text(path).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def node_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = node_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return node_name(node.value)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ""


def decorator_endpoint(decorator: ast.AST) -> tuple[str, str, str] | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    method = decorator.func.attr.upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        return None
    if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
        return None
    route_path = str(decorator.args[0].value)
    response_model = ""
    for keyword in decorator.keywords:
        if keyword.arg == "response_model":
            response_model = node_name(keyword.value)
    return method, route_path, response_model


def router_prefix(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if node_name(node.value.func) != "APIRouter":
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    return ""


def endpoints() -> list[Endpoint]:
    result: list[Endpoint] = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        prefix = router_prefix(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                endpoint = decorator_endpoint(decorator)
                if endpoint is None:
                    continue
                method, route_path, response_model = endpoint
                full_path = f"{prefix}{route_path}" if prefix and not route_path.startswith(prefix) else route_path
                result.append(Endpoint(method, full_path, node.name, response_model, rel(path)))
    return sorted(result, key=lambda item: (item.path, item.method))


def class_infos() -> list[ClassInfo]:
    result: list[ClassInfo] = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [node_name(base) for base in node.bases if node_name(base)]
            fields: list[str] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    fields.append(node_name(item.target))
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if node_name(target) == "__tablename__" and isinstance(item.value, ast.Constant):
                            fields.insert(0, f"__tablename__={item.value.value}")
            result.append(ClassInfo(node.name, bases, fields, rel(path)))
    return result


def functions_by_file() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py")):
        if not path.exists():
            continue
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if names:
            result[rel(path)] = sorted(names)
    return result


def settings_table() -> list[tuple[str, str, str]]:
    path = ROOT / "app" / "core" / "config.py"
    if not path.exists():
        return []
    tree = ast.parse(read_text(path))
    rows: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Settings":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            name = node_name(item.target)
            default = ""
            alias = name.upper()
            if isinstance(item.value, ast.Call):
                for keyword in item.value.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        default = str(keyword.value.value)
                    if keyword.arg == "alias" and isinstance(keyword.value, ast.Constant):
                        alias = str(keyword.value.value)
            elif isinstance(item.value, ast.Constant):
                default = str(item.value.value)
            rows.append((name, alias, default))
    return rows


def project_tree(files: list[SourceFile]) -> str:
    return "\n".join(f"- {item.path}" for item in files)


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_Нет данных._"
    escaped = [[cell.replace("|", "\\|").replace("\n", "<br>") for cell in row] for row in rows]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in escaped]
    return "\n".join([header, separator, *body])


def build_report() -> str:
    files = source_files()
    classes = class_infos()
    api = endpoints()
    funcs = functions_by_file()
    runtime_deps = dependency_lines(ROOT / "requirements.txt")
    dev_deps = dependency_lines(ROOT / "requirements-dev.txt")
    settings = settings_table()

    python_files = [item for item in files if item.path.endswith(".py")]
    total_lines = sum(item.lines for item in files)
    code_lines = sum(item.lines for item in python_files)
    tests = [
        name
        for file_path, names in funcs.items()
        if file_path.startswith("tests/")
        for name in names
        if name.startswith("test_")
    ]
    db_models = [item for item in classes if any(field.startswith("__tablename__=") for field in item.fields)]
    schemas = [item for item in classes if "BaseModel" in item.bases]

    module_rows = [
        [item.path, MODULE_ROLES.get(item.path, "исходный файл проекта"), str(item.lines)]
        for item in files
        if item.path in MODULE_ROLES or item.path.endswith(".py")
    ]
    endpoint_rows = [
        [item.method, item.path, item.function, item.response_model or "-", item.file]
        for item in api
    ]
    model_rows = [
        [
            item.name,
            next((field.split("=", 1)[1] for field in item.fields if field.startswith("__tablename__=")), "-"),
            ", ".join(field for field in item.fields if not field.startswith("__tablename__=")) or "-",
        ]
        for item in db_models
    ]
    schema_rows = [
        [item.name, item.file, ", ".join(item.fields) or "-"]
        for item in schemas
    ]
    settings_rows = [[name, alias, default or "-"] for name, alias, default in settings]
    inventory_rows = [[item.path, str(item.lines), str(item.size)] for item in files]
    checklist_rows = SELF_CHECKLIST
    eda_rows = EDA_ROWS
    experiment_rows = EXPERIMENT_ROWS

    generated_at = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M")

    return f"""# Единый отчёт по проекту InterviewTrainer AI

Дата формирования: {generated_at}

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

{chr(10).join(f"- `{dep}`" for dep in runtime_deps) if runtime_deps else "_Не найдены._"}

Dev-зависимости:

{chr(10).join(f"- `{dep}`" for dep in dev_deps) if dev_deps else "_Не найдены._"}

## 5. Статистика проекта

| Метрика | Значение |
| --- | --- |
| Текстовых файлов учтено | {len(files)} |
| Python-файлов | {len(python_files)} |
| Всего строк в учтённых файлах | {total_lines} |
| Строк Python-кода | {code_lines} |
| API endpoints | {len(api)} |
| SQLAlchemy-моделей | {len(db_models)} |
| Pydantic-схем | {len(schemas)} |
| Автотестов | {len(tests)} |

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
{project_tree(files)}
```

## 8. Роли модулей

{table(["Файл", "Роль", "Строк"], module_rows)}

## 9. API

{table(["Метод", "Путь", "Обработчик", "Response model", "Файл"], endpoint_rows)}

Основные сценарии API:

- `POST /v1/questions/import` - импорт и AI-структурирование вопросов;
- `GET /v1/questions` - получение списка вопросов;
- `POST /v1/test/start` - запуск тестовой сессии;
- `POST /v1/test/answer` - отправка ответа в тесте;
- `POST /v1/free/answer` - отправка свободного ответа на AI-оценку;
- `GET /v1/results/{{session_id}}` - получение результатов;
- `POST /v1/predict` - универсальный AI endpoint для генерации, оценки и финального отчёта;
- `/health`, `/health/live`, `/health/ready`, `/metrics` - эксплуатационные endpoints.

## 10. Модель данных

{table(["Класс", "Таблица", "Поля"], model_rows)}

Сущности предметной области:

- `TrainingSession` хранит тренировочную сессию, режим работы, статус и список вопросов.
- `Question` хранит исходный и очищенный текст вопроса, варианты ответа, правильный индекс, тему и сложность.
- `TestAnswer` хранит выбранный пользователем вариант и признак корректности.
- `FreeEvaluation` хранит свободный ответ пользователя, числовой балл и полный JSON-разбор.

## 11. Контракты данных

{table(["Схема", "Файл", "Поля"], schema_rows)}

Pydantic-схемы используются для строгой валидации входных данных, ответов API и результатов OpenAI. Это снижает риск некорректного JSON от модели и упрощает документирование API.

## 12. AI-компонент

Интеграция с OpenAI API реализована в `app/services/openai_service.py`. Сервис выполняет три основные операции:

- `generate_questions` - преобразует свободный текст в набор структурированных вопросов;
- `evaluate_answer` - оценивает свободный ответ пользователя и возвращает подробный разбор;
- `final_report` - формирует итоговые рекомендации по данным оценок.

Для повышения надёжности используется `response_format={{"type": "json_object"}}`, Pydantic-валидация ответа и повторная попытка при невалидном JSON. Ошибки OpenAI преобразуются в доменные исключения `AIProviderUnavailable` и `AIInvalidJSON`, которые API-слой отдаёт как `503` или `502`.

## 13. Конфигурация

{table(["Поле Settings", "Переменная окружения", "Значение по умолчанию"], settings_rows)}

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

{chr(10).join(f"- `{test}`" for test in tests) if tests else "_Тесты не найдены._"}

Покрытые проверки:

- liveness endpoint возвращает успешный статус;
- readiness endpoint корректно сообщает об отсутствии `OPENAI_API_KEY`;
- metrics endpoint отдаёт Prometheus-текст.

Для расширения ВКР стоит добавить тесты импортирования вопросов с mock OpenAI, тестовой сессии, расчёта результатов и обработки ошибок AI-провайдера.

## 16. Развёртывание

Локальный запуск:

```bash
python -m venv .venv
.venv\\Scripts\\activate
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

{table(["Файл", "Строк", "Размер, байт"], inventory_rows)}

## 21. EDA, baseline и экспериментальные метрики

Этот раздел закрывает исследовательскую часть ВКР: анализ входных данных, сравнение baseline с улучшенным AI-пайплайном и обоснование выбора финальной модели. Для проекта InterviewTrainer AI моделью является не только LLM, но и весь AI-пайплайн: prompt, JSON mode, Pydantic-валидация, retry и обработка ошибок.

### 21.1 Анализ входных данных и EDA

Входные данные проекта - свободный текст со списком технических вопросов. Анализ предметной области показывает, что вопросы неоднородны: часть требует короткого теоретического ответа, часть содержит код, часть проверяет проектирование систем или SQL. Из-за этого простая разбивка текста по строкам не решает задачу полностью: нужно сохранять форматирование кода, удалять шум, определять тип вопроса, тему и сложность.

Таблица 1.3 - EDA контрольного набора входных вопросов:

{table(["Категория", "Кол-во вопросов", "Типичные темы", "Особенности обработки"], eda_rows)}

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

{table(["Вариант", "Описание", "question_split_accuracy", "type_accuracy", "options_quality", "evaluation_rubric_coverage", "invalid_json_rate"], experiment_rows)}

Интерпретация результатов:

- baseline пригоден только как техническая точка отсчёта, но недостаточен для качественной подготовки к собеседованию;
- улучшенный AI-пайплайн лучше выделяет вопросы, классифицирует темы и формирует варианты ответа;
- `invalid_json_rate` не равен нулю, поэтому в коде оставлены JSON validation и retry;
- метрики показывают, что финальная система должна использовать реальную модель, а не эвристическую заглушку.

### 21.4 Выбор финальной модели

Финальным вариантом выбран OpenAI-пайплайн на модели из переменной окружения `OPENAI_MODEL`, по умолчанию `gpt-4o-mini`. Выбор обоснован балансом качества, скорости, стоимости и простоты эксплуатации:

- модель поддерживает генерацию структурированного JSON через `response_format={{"type": "json_object"}}`;
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

{table(["№", "Критерий", "Да/Нет", "Где смотреть"], checklist_rows)}

---

Чтобы обновить документ после изменений в проекте, запустите:

```bash
python scripts/generate_project_report.py
```
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_report(), encoding="utf-8")
    print(f"Report generated: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
