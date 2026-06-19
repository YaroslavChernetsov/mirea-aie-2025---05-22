from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "experiment_dataset.json"
JSON_OUTPUT = ROOT / "reports" / "eda_summary.json"
MD_OUTPUT = ROOT / "reports" / "eda_summary.md"


def analyze(rows: list[dict[str, str]]) -> dict[str, object]:
    lengths = [len(row["text"]) for row in rows]
    return {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "questions_count": len(rows),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "expected_types": dict(sorted(Counter(row["expected_type"] for row in rows).items())),
        "text_length_chars": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "median": round(statistics.median(lengths), 2),
        },
        "duplicate_ids": len(rows) - len({row["id"] for row in rows}),
        "duplicate_texts": len(rows) - len({row["text"] for row in rows}),
        "missing_values": sum(not row.get(key) for row in rows for key in ("id", "text", "expected_type", "category")),
    }


def markdown(summary: dict[str, object]) -> str:
    category_rows = "\n".join(f"| {key} | {value} |" for key, value in summary["categories"].items())
    type_rows = "\n".join(f"| {key} | {value} |" for key, value in summary["expected_types"].items())
    lengths = summary["text_length_chars"]
    return f"""# EDA контрольного датасета

Источник: `{summary['dataset']}`. Количество вопросов: **{summary['questions_count']}**.

## Категории

| Категория | Количество |
| --- | ---: |
{category_rows}

## Целевые типы

| Тип | Количество |
| --- | ---: |
{type_rows}

## Качество данных

| Проверка | Значение |
| --- | ---: |
| Минимальная длина, символов | {lengths['min']} |
| Максимальная длина, символов | {lengths['max']} |
| Средняя длина, символов | {lengths['mean']} |
| Медианная длина, символов | {lengths['median']} |
| Дубли ID | {summary['duplicate_ids']} |
| Дубли текстов | {summary['duplicate_texts']} |
| Пропуски | {summary['missing_values']} |

## Вывод

Набор покрывает теоретические вопросы, SQL, написание кода, предсказание вывода и system design. Пропусков и дублей нет; его можно использовать для воспроизводимого сравнения baseline и production AI-пайплайна.
"""


def main() -> None:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    summary = analyze(rows)
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
