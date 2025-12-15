import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _build_data_paths(base_dir: str) -> tuple[str, str]:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    app_path = os.path.join(data_dir, "application_.parquet")
    bureau_path = os.path.join(data_dir, "bureau.parquet")
    return app_path, bureau_path


def _validate_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro: {path}")


def _select_numeric_columns(df: pd.DataFrame, limit: int = 25) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) <= limit:
        return numeric_cols
    variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    return variances.head(limit).index.tolist()


def _plot_target_distribution(df: pd.DataFrame, output_path: str) -> None:
    if "TARGET" not in df.columns:
        print("No se genera grafico de TARGET: columna 'TARGET' no existe.")
        return

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="TARGET")
    ax.set_title("Distribucion del Target (TARGET)")
    ax.set_xlabel("TARGET")
    ax.set_ylabel("Conteo")

    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height()):,}".replace(",", "."),
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def _plot_correlation_matrix(df: pd.DataFrame, output_path: str, max_features: int = 25) -> None:
    numeric_cols = _select_numeric_columns(df, limit=max_features)
    if len(numeric_cols) < 2:
        print("No se genera correlacion: no hay suficientes columnas numericas.")
        return

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2, cbar=True)
    plt.title(f"Matriz de Correlacion Basica (top {len(numeric_cols)} numericas)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def main() -> None:
    sns.set_theme(style="whitegrid")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path, bureau_path = _build_data_paths(current_dir)

    print("=== [1] Cargando application_.parquet ===")
    _validate_file(app_path)
    app = pd.read_parquet(app_path)

    print("\n--- application_ ---")
    print(f"Shape (filas, columnas): {app.shape}")
    print("\nTipos de datos (dtypes):")
    print(app.dtypes.sort_values())

    if "TARGET" not in app.columns:
        print("\n[WARN] No se encontro la columna 'TARGET' en application_.parquet.")
    else:
        print("\nDistribucion de TARGET (conteo):")
        print(app["TARGET"].value_counts(dropna=False))
        print("\nDistribucion de TARGET (porcentaje):")
        print(app["TARGET"].value_counts(normalize=True, dropna=False).round(4))

    print("\n=== [2] Cargando bureau.parquet (tabla secundaria) ===")
    _validate_file(bureau_path)
    bureau = pd.read_parquet(bureau_path)

    print("\n--- bureau ---")
    print(f"Shape (filas, columnas): {bureau.shape}")

    nulls = bureau.isna().sum().sort_values(ascending=False)
    nulls_pct = (nulls / len(bureau)).replace([float("inf")], 0) * 100
    null_report = pd.DataFrame({"n_nulls": nulls, "pct_nulls": nulls_pct.round(2)})

    print("\nTop 20 columnas con mas nulos en bureau:")
    print(null_report.head(20).to_string())

    print("\n=== [3-4] Generando y guardando graficos ===")
    target_plot_path = os.path.join(OUT_DIR, "01_target_distribution.png")
    corr_plot_path = os.path.join(OUT_DIR, "02_correlation_matrix.png")

    _plot_target_distribution(app, target_plot_path)
    _plot_correlation_matrix(app, corr_plot_path, max_features=25)

    print("\nFase 1 (Data Understanding) completada.")


if __name__ == "__main__":
    main()
