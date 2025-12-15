import json
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD_RANGE = np.arange(0.05, 0.951, 0.01)


def ensure_numeric_no_na(df: pd.DataFrame) -> None:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            "Se esperaban solo columnas numericas despues del OneHot. "
            f"Columnas no numericas detectadas (primeras 10): {non_numeric[:10]}"
        )
    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Se encontraron NaNs en X. Columnas con NaNs (primeras 10): {na_cols[:10]}")


def load_artifacts(base_dir: str) -> Tuple[Any, pd.DataFrame, pd.Series]:
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    X_path = os.path.join(artifacts_dir, "X_train_processed.parquet")
    y_path = os.path.join(artifacts_dir, "y_train_processed.parquet")

    print("==> Cargando artefactos")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("No se encontraron X_train_processed.parquet o y_train_processed.parquet en /artifacts.")

    model = joblib.load(model_path)
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)["TARGET"]

    ensure_numeric_no_na(X)
    return model, X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("==> Realizando train_test_split (80/20 estratificado)")
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Estrategia:
    1) Buscar umbrales que alcancen Recall >= 0.70 maximizando Precision.
    2) Si ninguno cumple, usar el umbral con mejor F1.
    """
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "strategy": ""}

    recall_candidates = []
    for thr in THRESHOLD_RANGE:
        preds = (y_proba >= thr).astype(int)
        rec = recall_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds)
        recall_candidates.append((thr, prec, rec, f1))

    recall_valid = [c for c in recall_candidates if c[2] >= 0.70]
    if recall_valid:
        thr, prec, rec, f1 = max(recall_valid, key=lambda x: (x[1], x[3]))
        best.update({"threshold": thr, "precision": prec, "recall": rec, "f1": f1, "strategy": "Recall>=0.70 max Precision"})
    else:
        thr, prec, rec, f1 = max(recall_candidates, key=lambda x: x[3])
        best.update({"threshold": thr, "precision": prec, "recall": rec, "f1": f1, "strategy": "Max F1 (sin Recall>=0.70)"})

    print(
        f"==> Umbral optimo: {best['threshold']:.2f} | Estrategia: {best['strategy']} | "
        f"F1: {best['f1']:.4f} | Recall: {best['recall']:.4f} | Precision: {best['precision']:.4f}"
    )
    return best["threshold"], best


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, output_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true: pd.Series, y_proba: np.ndarray, output_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm_dict: Dict[str, int], output_path: str) -> None:
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred 0", "Pred 1"], yticklabels=["Real 0", "Real 1"])
    plt.title("Matriz de confusion")
    plt.ylabel("Real")
    plt.xlabel("Prediccion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: Any, feature_names: pd.Index, output_path: str, top_n: int = 25) -> None:
    plt.figure(figsize=(8, 10))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        names_top = feature_names[idx]
        imp_top = importances[idx]
        sns.barplot(x=imp_top, y=names_top, orient="h")
        plt.title(f"Feature Importance (Top {top_n})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "Feature importance no disponible", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_report(report_path: str, report: Dict[str, Any]) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    sns.set_theme(style="whitegrid")

    model, X, y = load_artifacts(BASE_DIR)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("El modelo cargado no expone predict_proba, requerido para evaluacion.")

    print("==> Generando probabilidades en holdout")
    y_proba = model.predict_proba(X_test)[:, 1]

    print("==> Buscando umbral optimo")
    best_threshold, threshold_info = find_best_threshold(y_test, y_proba)

    print("==> Calculando metricas finales")
    y_pred = (y_proba >= best_threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("==> Generando clasificacion detallada (opcional)")
    classif_report = classification_report(y_test, y_pred, digits=4)

    print("==> Generando graficos")
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    pr_path = os.path.join(OUT_DIR, "pr_curve.png")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fi_path = os.path.join(OUT_DIR, "feature_importance.png")

    plot_roc_curve(y_test, y_proba, roc_path)
    plot_pr_curve(y_test, y_proba, pr_path)
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
    plot_feature_importance(model, X.columns, fi_path)

    report = {
        "threshold": float(best_threshold),
        "threshold_selection": threshold_info,
        "metrics": metrics,
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
        },
        "notes": {
            "threshold_strategy": "Primero busca Recall>=0.70 maximizando Precision; si no existe, usa el mejor F1.",
            "metrics": "ROC-AUC, PR-AUC, Accuracy (referencial), Precision, Recall, F1, matriz de confusion.",
        },
    }

    report_path = os.path.join(OUT_DIR, "evaluation_report.json")
    print(f"==> Guardando reporte en: {report_path}")
    save_report(report_path, report)

    cr_path = os.path.join(OUT_DIR, "classification_report.txt")
    with open(cr_path, "w", encoding="utf-8") as f:
        f.write(classif_report)

    print("\nFase 4 (Evaluation) completada.")


if __name__ == "__main__":
    main()
