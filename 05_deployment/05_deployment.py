import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent 
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
COLUMNS_PATH = ARTIFACTS_DIR / "X_columns.json"
EVAL_REPORT_PATH = BASE_DIR / "evaluation_report.json"
DEFAULT_THRESHOLD = 0.50

app = FastAPI(
    title="Credit Default Risk API",
    version="1.0.0",
    description="API para predecir probabilidad de incumplimiento (Home Credit).",
)

model: Optional[Any] = None
expected_columns: Optional[list[str]] = None
threshold_high: float = DEFAULT_THRESHOLD
threshold_low: float = round(DEFAULT_THRESHOLD * 0.65, 2)


class PredictRequest(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description="Diccionario con features del cliente: {columna: valor, ...}",
    )


def is_valid_threshold(val: Any) -> bool:
    return isinstance(val, (int, float)) and 0 <= float(val) <= 1


def extract_threshold_from_dict(d: Dict[str, Any]) -> Optional[float]:
    keys_order = ["threshold_optimo", "best_threshold", "optimal_threshold", "threshold"]
    for k in keys_order:
        if k in d and is_valid_threshold(d[k]):
            return float(d[k])
    for key in ["best", "metrics", "evaluation", "report"]:
        if key in d and isinstance(d[key], dict):
            t = extract_threshold_from_dict(d[key])
            if t is not None:
                return t
    return None


def load_threshold() -> float:
    if not EVAL_REPORT_PATH.exists():
        return DEFAULT_THRESHOLD
    try:
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        if isinstance(report, dict):
            t = extract_threshold_from_dict(report)
            if t is not None:
                return t
    except Exception:
        return DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def ensure_numeric(df: pd.DataFrame) -> None:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Columnas no numéricas detectadas: {non_numeric[:10]}")


def build_input_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if expected_columns is not None:
        df = df.reindex(columns=expected_columns, fill_value=0)

    ensure_numeric(df)
    return df


def load_columns(path: Path) -> Optional[list[str]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    return list(cols) if cols is not None else None


def load_model(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"No se encontró el modelo en: {path}")
    return joblib.load(path)


@app.on_event("startup")
def load_artifacts() -> None:
    global model, expected_columns, threshold_high, threshold_low
    model = load_model(MODEL_PATH)
    expected_columns = load_columns(COLUMNS_PATH)
    threshold_high = load_threshold()
    threshold_low = round(threshold_high * 0.65, 2)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "has_expected_columns": expected_columns is not None,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    try:
        df = build_input_dataframe(req.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input inválido: {exc}")

    try:
        if hasattr(model, "predict_proba"):
            proba_default = float(model.predict_proba(df)[:, 1][0])
        else:
            score = float(model.predict(df)[0])
            proba_default = float(1 / (1 + np.exp(-score)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}")

    if proba_default < threshold_low:
        decision = "Aprobado"
    elif proba_default < threshold_high:
        decision = "Revisión Manual"
    else:
        decision = "Rechazado"

    return {
        "probabilidad_default": round(proba_default, 6),
        "decision": decision,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
    }


# -------------------------------------------------
# Ejecución:
# python -m uvicorn 05_deployment:app --reload --host 127.0.0.1 --port 8000
#
# Ejemplo /predict:
# curl -X POST "http://127.0.0.1:8000/predict" ^
#   -H "Content-Type: application/json" ^
#   -d "{\"data\": {\"FEATURE_1\": 0.1, \"FEATURE_2\": 3}}"
# -------------------------------------------------
