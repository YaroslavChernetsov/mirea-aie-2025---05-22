import json
from pathlib import Path

from scripts.run_eda import analyze
from scripts.run_experiment import baseline_predictions, metrics


ROOT = Path(__file__).resolve().parents[1]


def dataset() -> list[dict[str, str]]:
    return json.loads((ROOT / "data" / "experiment_dataset.json").read_text(encoding="utf-8"))


def test_eda_dataset_has_no_duplicates_or_missing_values() -> None:
    summary = analyze(dataset())
    assert summary["questions_count"] >= 10
    assert summary["duplicate_ids"] == 0
    assert summary["duplicate_texts"] == 0
    assert summary["missing_values"] == 0
    assert len(summary["expected_types"]) >= 5


def test_baseline_experiment_metrics_are_bounded() -> None:
    rows = dataset()
    result = metrics(rows, baseline_predictions(rows))
    assert result["samples"] == len(rows)
    for name in ("question_split_accuracy", "type_accuracy", "options_quality_rate", "schema_valid_rate"):
        assert 0 <= result[name] <= 1
    assert result["options_quality_rate"] == 0


def test_saved_production_experiment_beats_baseline_options_quality() -> None:
    rows = dataset()
    snapshot = json.loads((ROOT / "reports" / "experiment_predictions.json").read_text(encoding="utf-8"))
    baseline = metrics(rows, baseline_predictions(rows))
    improved = metrics(rows, snapshot["predictions"])
    assert improved["predictions"] == len(rows)
    assert improved["schema_valid_rate"] == 1
    assert improved["options_quality_rate"] > baseline["options_quality_rate"]
