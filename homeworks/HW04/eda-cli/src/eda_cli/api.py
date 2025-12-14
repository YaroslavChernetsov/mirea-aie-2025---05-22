# src/eda_cli/api.py
from __future__ import annotations

import io
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .core import summarize_dataset, missing_table, compute_quality_flags

app = FastAPI(
    title="EDA CLI — HTTP API",
    description="HTTP-обертка над eda-cli для анализа качества данных",
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Простая проверка работоспособности сервиса."""
    return {"status": "ok", "message": "EDA API is running"}


@app.post("/quality")
async def quality_from_dataframe(data: Dict[str, Any]) -> JSONResponse:
    """
    Принимает JSON вида {"columns": {"col1": [...], "col2": [...]}},
    строит DataFrame и возвращает флаги качества.
    """
    try:
        df = pd.DataFrame(data["columns"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty dataset")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    return JSONResponse(content=flags)


@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)) -> JSONResponse:
    """
    Принимает CSV-файл, читает его и возвращает флаги качества.
    """
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty dataset")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    return JSONResponse(content=flags)