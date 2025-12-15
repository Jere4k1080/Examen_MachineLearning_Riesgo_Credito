import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)



def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace({0: np.nan})


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_prefix(df: pd.DataFrame, prefix: str, exclude_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    exclude_cols = set(exclude_cols or [])
    new_cols: Dict[str, str] = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        new_cols[col] = f"{prefix}__{col}"
    return df.rename(columns=new_cols)


def sanitize_columns(columns: List[str]) -> List[str]:
    """Reemplaza caracteres no [A-Za-z0-9_] por _, colapsa _, y garantiza unicidad."""
    sanitized = []
    seen = {}
    for col in columns:
        c = re.sub(r"[^A-Za-z0-9_]", "_", col)
        c = re.sub(r"_+", "_", c).strip("_")
        if not c:
            c = "col"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        sanitized.append(c)
    return sanitized



def agg_bureau(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "bureau.parquet")
    bureau = pd.read_parquet(path)
    bureau_num = bureau.select_dtypes(include=["number"]).copy()

    num_aggs = {}
    for col in bureau_num.columns:
        if col in ["SK_ID_CURR", "SK_ID_BUREAU"]:
            continue
        num_aggs[col] = ["mean", "max", "min", "sum", "std"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(num_aggs)
    bureau_agg.columns = ["_".join([c[0], c[1]]).upper() for c in bureau_agg.columns]
    bureau_agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR")["SK_ID_BUREAU"].count()

    bureau_agg = bureau_agg.reset_index()
    bureau_agg = add_prefix(bureau_agg, "BUREAU", exclude_cols=["SK_ID_CURR"])
    return bureau_agg


def agg_bureau_balance(data_dir: str) -> pd.DataFrame:
    bb_path = os.path.join(data_dir, "bureau_balance.parquet")
    b_path = os.path.join(data_dir, "bureau.parquet")

    bb = pd.read_parquet(bb_path)
    bureau = pd.read_parquet(b_path)[["SK_ID_BUREAU", "SK_ID_CURR"]]

    status_counts = pd.crosstab(bb["SK_ID_BUREAU"], bb["STATUS"])
    status_counts.columns = [f"STATUS_{str(c)}" for c in status_counts.columns]
    status_counts = status_counts.reset_index()

    bb_num = bb.select_dtypes(include=["number"]).copy()
    bb_agg_num = bb.groupby("SK_ID_BUREAU").agg({"MONTHS_BALANCE": ["min", "max", "mean", "count"]})
    bb_agg_num.columns = ["_".join([c[0], c[1]]).upper() for c in bb_agg_num.columns]
    bb_agg_num = bb_agg_num.reset_index()

    bb_agg = bb_agg_num.merge(status_counts, on="SK_ID_BUREAU", how="left")
    bb_agg = bb_agg.merge(bureau, on="SK_ID_BUREAU", how="left")

    agg_dict = {c: ["sum"] for c in bb_agg.columns if c.startswith("STATUS_")}
    agg_dict.update(
        {
            "MONTHS_BALANCE_MIN": ["min"],
            "MONTHS_BALANCE_MAX": ["max"],
            "MONTHS_BALANCE_MEAN": ["mean"],
            "MONTHS_BALANCE_COUNT": ["sum"],
        }
    )

    client_agg = bb_agg.groupby("SK_ID_CURR").agg(agg_dict)
    client_agg.columns = ["_".join([c[0], c[1]]).upper() for c in client_agg.columns]
    client_agg = client_agg.reset_index()
    client_agg = add_prefix(client_agg, "BUREAU_BAL", exclude_cols=["SK_ID_CURR"])
    return client_agg


def agg_previous_application(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "previous_application.parquet")
    prev = pd.read_parquet(path)

    prev_num = prev.select_dtypes(include=["number"]).copy()
    num_aggs = {}
    for col in prev_num.columns:
        if col in ["SK_ID_CURR", "SK_ID_PREV"]:
            continue
        num_aggs[col] = ["mean", "max", "min", "sum", "std"]

    prev_agg = prev.groupby("SK_ID_CURR").agg(num_aggs)
    prev_agg.columns = ["_".join([c[0], c[1]]).upper() for c in prev_agg.columns]
    prev_agg["PREVAPP_COUNT"] = prev.groupby("SK_ID_CURR")["SK_ID_PREV"].count()

    prev_agg = prev_agg.reset_index()
    prev_agg = add_prefix(prev_agg, "PREVAPP", exclude_cols=["SK_ID_CURR"])
    return prev_agg


def agg_pos_cash_balance(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "POS_CASH_balance.parquet")
    pos = pd.read_parquet(path)

    pos_num = pos.select_dtypes(include=["number"]).copy()
    aggs = {}
    for col in pos_num.columns:
        if col in ["SK_ID_CURR", "SK_ID_PREV"]:
            continue
        aggs[col] = ["mean", "max", "min", "sum", "std"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggs)
    pos_agg.columns = ["_".join([c[0], c[1]]).upper() for c in pos_agg.columns]
    pos_agg["POS_COUNT_PREV"] = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    pos_agg["POS_COUNT_ROWS"] = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].count()

    pos_agg = pos_agg.reset_index()
    pos_agg = add_prefix(pos_agg, "POS", exclude_cols=["SK_ID_CURR"])
    return pos_agg


def agg_installments_payments(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "installments_payments.parquet")
    ins = pd.read_parquet(path)

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(ins.columns):
        ins["PAYMENT_PERC"] = safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(ins.columns):
        ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
        ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    agg_dict = {}
    if "AMT_PAYMENT" in ins.columns:
        agg_dict["AMT_PAYMENT"] = ["mean", "var", "max", "sum"]

    for col in ["PAYMENT_PERC", "PAYMENT_DIFF", "DPD", "DBD"]:
        if col in ins.columns:
            agg_dict[col] = ["mean", "max", "sum", "var"]

    ins_agg = ins.groupby("SK_ID_CURR").agg(agg_dict)
    ins_agg.columns = ["_".join([c[0], c[1]]).upper() for c in ins_agg.columns]
    ins_agg["INS_COUNT_ROWS"] = ins.groupby("SK_ID_CURR").size()
    if "SK_ID_PREV" in ins.columns:
        ins_agg["INS_COUNT_PREV"] = ins.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()

    ins_agg = ins_agg.reset_index()
    ins_agg = add_prefix(ins_agg, "INS", exclude_cols=["SK_ID_CURR"])
    return ins_agg


def agg_credit_card_balance(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "credit_card_balance.parquet")
    ccb = pd.read_parquet(path)

    ccb_num = ccb.select_dtypes(include=["number"]).copy()
    aggs = {}
    for col in ccb_num.columns:
        if col in ["SK_ID_CURR", "SK_ID_PREV"]:
            continue
        aggs[col] = ["mean", "max", "min", "sum", "std"]

    ccb_agg = ccb.groupby("SK_ID_CURR").agg(aggs)
    ccb_agg.columns = ["_".join([c[0], c[1]]).upper() for c in ccb_agg.columns]
    ccb_agg["CCB_COUNT_PREV"] = ccb.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    ccb_agg["CCB_COUNT_ROWS"] = ccb.groupby("SK_ID_CURR").size()

    ccb_agg = ccb_agg.reset_index()
    ccb_agg = add_prefix(ccb_agg, "CCB", exclude_cols=["SK_ID_CURR"])
    return ccb_agg


def impute_simple(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    for c in cat_cols:
        if df[c].isna().all():
            df[c] = df[c].fillna("MISSING")
        else:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "MISSING"
            df[c] = df[c].fillna(fill)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for c in cat_cols:
        df[c] = df[c].astype("category")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df_encoded



def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    ensure_dir(artifacts_dir)

    app_path = os.path.join(data_dir, "application_.parquet")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"No se encontro application_.parquet en: {app_path}")

    app = pd.read_parquet(app_path)
    if "TARGET" not in app.columns:
        raise ValueError("La tabla application_ no contiene la columna TARGET (requerida).")

    train = app[app["TARGET"].notna()].copy()
    y = train["TARGET"].astype(int)
    X = train.drop(columns=["TARGET"])

    print("==> Generando agregaciones 1:N (secundarias -> cliente)")
    features: List[pd.DataFrame] = [X]
    exists = lambda name: os.path.exists(os.path.join(data_dir, name))

    if exists("bureau.parquet"):
        features.append(agg_bureau(data_dir))
    if exists("bureau_balance.parquet") and exists("bureau.parquet"):
        features.append(agg_bureau_balance(data_dir))
    if exists("previous_application.parquet"):
        features.append(agg_previous_application(data_dir))
    if exists("POS_CASH_balance.parquet"):
        features.append(agg_pos_cash_balance(data_dir))
    if exists("installments_payments.parquet"):
        features.append(agg_installments_payments(data_dir))
    if exists("credit_card_balance.parquet"):
        features.append(agg_credit_card_balance(data_dir))

    print("==> Uniendo features agregadas a application_ (por SK_ID_CURR)")
    df = features[0]
    for f in features[1:]:
        df = df.merge(f, on="SK_ID_CURR", how="left")

    print("==> Imputacion simple (mediana/mode/MISSING)")
    df = impute_simple(df)

    print("==> OneHot Encoding (categoricas)")
    df = one_hot_encode(df)

    print("==> Sanitizando nombres de columnas")
    df.columns = sanitize_columns(df.columns.tolist())

    X_out = os.path.join(artifacts_dir, "X_train_processed.parquet")
    y_out = os.path.join(artifacts_dir, "y_train_processed.parquet")
    cols_out = os.path.join(artifacts_dir, "X_columns.json")

    print(f"==> Guardando X en: {X_out}")
    df.to_parquet(X_out, index=False)

    print(f"==> Guardando y en: {y_out}")
    y.to_frame("TARGET").to_parquet(y_out, index=False)

    print(f"==> Guardando columnas de X en: {cols_out}")
    with open(cols_out, "w", encoding="utf-8") as f:
        json.dump(list(df.columns), f, ensure_ascii=False, indent=2)

    print("\n Fase 2 (Data Preparation) completada.")
    print(f"X shape: {df.shape} | y shape: {y.shape}")


if __name__ == "__main__":
    main()
