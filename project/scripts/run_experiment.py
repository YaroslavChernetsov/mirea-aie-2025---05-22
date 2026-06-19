from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.openai_service import OpenAIService


DATASET = ROOT / "data" / "experiment_dataset.json"
SNAPSHOT = ROOT / "reports" / "experiment_predictions.json"
OUTPUT = ROOT / "reports" / "experiment_metrics.json"


def baseline_type(text: str) -> str:
    value = text.lower()
    if "sql" in value or "join" in value:
        return "sql"
    if any(token in value for token in ("спроектиру", "отказоустойчив", "дата-центр")):
        return "system_design"
    if any(token in value for token in ("что выведет", "output", "console.log")):
        return "output_prediction"
    if any(token in value for token in ("напишите", "реализуйте", "функци")):
        return "coding"
    return "theory"


def baseline_predictions(dataset: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "question_type": baseline_type(row["text"]),
            "options": ["Верно", "Неверно", "Зависит от контекста", "Недостаточно данных"],
        }
        for row in dataset
    ]


async def live_predictions(dataset: list[dict[str, str]]) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_configured:
        raise SystemExit("OPENAI_API_KEY is required for --live")
    service = OpenAIService(settings)
    raw_text = "\n".join(f"{index}. [{row['id']}] {row['text']}" for index, row in enumerate(dataset, 1))
    response = await service.generate_questions(raw_text, len(dataset))
    predictions = []
    for row, question in zip(dataset, response.questions, strict=False):
        predictions.append(
            {"id": row["id"], "question_type": question.question_type.value, "options": question.options}
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.openai_model,
        "predictions": predictions,
    }


def metrics(dataset: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    by_id = {row["id"]: row for row in predictions}
    matched = [(row, by_id[row["id"]]) for row in dataset if row["id"] in by_id]
    quality_options = 0
    generic_options = {"верно", "неверно", "зависит от контекста", "недостаточно данных"}
    for _, prediction in matched:
        options = [str(value).strip() for value in prediction.get("options", [])]
        normalized = {value.casefold() for value in options}
        quality_options += int(
            len(options) == 4
            and all(options)
            and len(normalized) == 4
            and not normalized.issubset(generic_options)
        )
    total = len(dataset)
    return {
        "samples": total,
        "predictions": len(matched),
        "question_split_accuracy": round(len(matched) / total, 4),
        "type_accuracy": round(sum(actual["expected_type"] == predicted["question_type"] for actual, predicted in matched) / total, 4),
        "options_quality_rate": round(quality_options / total, 4),
        "schema_valid_rate": round(len(matched) / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the heuristic baseline with the production AI pipeline.")
    parser.add_argument("--live", action="store_true", help="Call OpenAI and refresh the improved-pipeline snapshot.")
    args = parser.parse_args()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))

    if args.live:
        snapshot = asyncio.run(live_predictions(dataset))
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif SNAPSHOT.exists():
        snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    else:
        raise SystemExit("No improved-model snapshot. Run once with --live.")

    result = {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_generated_at": snapshot["generated_at"],
        "model": snapshot["model"],
        "baseline": metrics(dataset, baseline_predictions(dataset)),
        "improved": metrics(dataset, snapshot["predictions"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
