from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
    plot_categorical_bars,  
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Количество top-значений для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта в report.md."),
    min_missing_share: float = typer.Option(
        0.1,
        help="Порог доли пропусков (0.0–1.0), выше которого колонка считается проблемной.",
        min=0.0,
        max=1.0,
    ),
    json_summary: bool = typer.Option(False, "--json-summary", help="Сохранить summary.json с компактной сводкой."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    # Передаём top_k_categories в top_categories
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # Идентифицируем проблемные колонки по пропускам
    problematic_missing_cols = (
        missing_df[missing_df["missing_share"] >= min_missing_share].index.tolist()
        if not missing_df.empty
        else []
    )

    # Собираем все "проблемные" колонки по разным критериям
    problematic_columns = set(problematic_missing_cols)

    # Константные колонки
    constant_cols = [col.name for col in summary.columns if col.unique == 1 and col.non_null > 0]
    problematic_columns.update(constant_cols)

    # Категориальные с высокой кардинальностью
    high_card_cols = [
        col.name for col in summary.columns
        if (not col.is_numeric) and col.unique > 50 and col.non_null > 0
    ]
    problematic_columns.update(high_card_cols)

    # Числовые с низкой вариативностью
    low_var_cols = [
        col.name for col in summary.columns
        if col.is_numeric and col.std is not None and col.std < 1e-6 and col.non_null > 1
    ]
    problematic_columns.update(low_var_cols)

    problematic_columns = sorted(problematic_columns)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")
    plot_categorical_bars(df, out_root, top_k=top_k_categories, max_columns=5)

    # 4. Сохраняем JSON-сводку, если запрошено
    if json_summary:
        import json
        json_summary_data = {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "quality_score": round(quality_flags["quality_score"], 4),
            "problematic_columns": problematic_columns,
            "flags": {
                k: v for k, v in quality_flags.items()
                if k not in ("quality_score", "max_missing_share")
            }
        }
        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(json_summary_data, f, indent=2, ensure_ascii=False)

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Порог пропусков для 'проблемных' колонок: **{min_missing_share:.2%}**\n")
        if problematic_missing_cols:
            f.write(f"- Проблемные колонки по пропускам: `{', '.join(problematic_missing_cols)}`\n")
        else:
            f.write("- Проблемных колонок по пропускам не обнаружено.\n")
        f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n")
            f.write(f"- Использован порог пропусков: **{min_missing_share:.2%}** для выявления проблемных колонок.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"- Для категориальных признаков показано top-{top_k_categories} значений.\n")
            f.write("См. файлы в папке `top_categories/`.\n\n")
            f.write(f"- Графики распределения категорий: `bar_*.png`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"- Показано максимум **{max_hist_columns}** гистограмм для числовых колонок.\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки — передаём параметры
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()