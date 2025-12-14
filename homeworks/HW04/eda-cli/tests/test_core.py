from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


# Вспомогательная функция — должна быть ОПРЕДЕЛЕНА ДО тестов, которые её используют!
def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # Даже если age содержит NaN, height числовая → corr может быть пустой
    # Поэтому лучше проверить, что она существует
    assert isinstance(corr, pd.DataFrame)

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_new_heuristics():
    """
    Проверяет новые эвристики качества данных:
    - has_constant_columns
    - has_high_cardinality_categoricals
    - has_numeric_columns_with_low_variation
    """
    n_rows = 51  # > 50 → превышает порог HIGH_CARDINALITY_THRESHOLD

    df = pd.DataFrame({
        "const_num": [42] * n_rows,
        "const_cat": ["A"] * n_rows,
        "high_card_cat": [f"id_{i}" for i in range(n_rows)],  # 51 уникальных
        "low_var_num": [1.0 + i * 1e-7 for i in range(n_rows)],  # std будет < 1e-6
        "normal_col": list(range(n_rows))
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Проверка флагов
    assert flags["has_constant_columns"] is True
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["has_numeric_columns_with_low_variation"] is True

    # Quality score должен быть снижен
    assert flags["quality_score"] < 1.0