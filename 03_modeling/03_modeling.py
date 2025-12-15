import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError("LightGBM es requerido. Instala con: pip install lightgbm") from exc

RANDOM_STATE = 42
MAX_RF_CV_ROWS = 80_000


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_all_numeric(df: pd.DataFrame) -> None:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            "Se esperaban solo columnas numericas despues del OneHot. "
            f"Columnas no numericas detectadas (primeras 10): {non_numeric[:10]}"
        )


def compute_scale_pos_weight(y: pd.Series) -> float:
    y = pd.Series(y).astype(int)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    train_size = max_rows / len(X)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def evaluate_cv(
    model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, model_name: str
) -> Dict[str, Any]:
    aucs: List[float] = []
    pr_aucs: List[float] = []
    f1s: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []

    for _, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model.fit(X_tr, y_tr)

        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"{model_name} no expone predict_proba, requerido para metricas.")

        proba = model.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, proba))
        pr_aucs.append(average_precision_score(y_va, proba))
        f1s.append(f1_score(y_va, pred))
        recalls.append(recall_score(y_va, pred))
        precisions.append(precision_score(y_va, pred, zero_division=0))

    return {
        "model": model_name,
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "f1_mean": float(np.mean(f1s)),
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "folds": [
            {
                "fold": i + 1,
                "roc_auc": float(aucs[i]),
                "pr_auc": float(pr_aucs[i]),
                "f1": float(f1s[i]),
                "recall": float(recalls[i]),
                "precision": float(precisions[i]),
            }
            for i in range(len(aucs))
        ],
    }


def main() -> None:
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    ensure_dir(artifacts_dir)

    X_path = os.path.join(artifacts_dir, "X_train_processed.parquet")
    y_path = os.path.join(artifacts_dir, "y_train_processed.parquet")

    print("==> Cargando datos procesados")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("No se encontraron X_train_processed.parquet o y_train_processed.parquet en /artifacts.")

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)["TARGET"]
    check_all_numeric(X)

    print("==> Train/Test split (80/20 estratificado)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print("==> Configurando CV (StratifiedKFold)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("==> Calculando scale_pos_weight en y_train")
    spw = compute_scale_pos_weight(y_train)
    print(f"    scale_pos_weight (neg/pos): {spw:.3f}")

    rf_params = {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    print("==> Submuestreando train para CV de RandomForest (baseline por costo computacional)")
    X_rf_cv, y_rf_cv = stratified_sample(X_train, y_train, max_rows=MAX_RF_CV_ROWS)
    print(f"    Tamaño muestra RF CV: {len(X_rf_cv)} filas")
    rf_model_cv = RandomForestClassifier(**rf_params)
    rf_cv_results = evaluate_cv(rf_model_cv, X_rf_cv, y_rf_cv, cv, "RandomForest_baseline")
    print("    Resultados CV RF:", rf_cv_results)

    lgbm_params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": spw,
    }
    print("==> CV LightGBM")
    lgbm_model_cv = LGBMClassifier(**lgbm_params)
    lgbm_cv_results = evaluate_cv(lgbm_model_cv, X_train, y_train, cv, "LightGBM_champion")
    print("    Resultados CV LGBM:", lgbm_cv_results)

    results_list = [rf_cv_results, lgbm_cv_results]
    best = max(results_list, key=lambda d: d["roc_auc_mean"])
    best_name = best["model"]
    print("\n==> Mejor modelo por ROC-AUC (CV):", best_name)
    print(best)

    if best_name.startswith("RandomForest"):
        best_model = RandomForestClassifier(**rf_params)
    else:
        best_model = LGBMClassifier(**lgbm_params)

    print("\n==> Entrenando mejor modelo en TODO el train")
    best_model.fit(X_train, y_train)

    if not hasattr(best_model, "predict_proba"):
        raise AttributeError("El mejor modelo no expone predict_proba, requerido para metricas.")

    print("==> Evaluando en holdout (X_test)")
    proba_test = best_model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    test_metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
        "test_pr_auc": float(average_precision_score(y_test, proba_test)),
        "test_f1": float(f1_score(y_test, pred_test)),
        "test_recall": float(recall_score(y_test, pred_test)),
        "test_precision": float(precision_score(y_test, pred_test, zero_division=0)),
    }
    print("    Metricas TEST:", test_metrics)

    model_out = os.path.join(artifacts_dir, "best_model.pkl")
    report_out = os.path.join(artifacts_dir, "model_report.json")

    print(f"\n==> Guardando modelo en: {model_out}")
    joblib.dump(best_model, model_out)

    report = {
        "cv_results": {
            "random_forest": rf_cv_results,
            "lightgbm": lgbm_cv_results,
        },
        "best_model": best,
        "test_metrics": test_metrics,
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
        },
        "notes": {
            "rf_sampling": f"RF CV limitado a máximo {MAX_RF_CV_ROWS} filas por costo computacional.",
            "imbalance_handling": "RF usa class_weight='balanced'. LGBM usa scale_pos_weight neg/pos calculado en y_train.",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "split": "train_test_split(test_size=0.20, stratify=y, random_state=42)",
        },
    }

    print(f"==> Guardando reporte en: {report_out}")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n Fase 3 (Modeling) completada.")


if __name__ == "__main__":
    main()
