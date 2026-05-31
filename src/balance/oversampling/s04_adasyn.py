"""
s04_adasyn.py
=============
ADASYN (Adaptive Synthetic Sampling): adapta la cantidad de muestras
sinteticas que genera por cada muestra de la clase minoritaria segun
lo dificil que sea de clasificar.  Muestras rodeadas de mayoria reciben
mas sinteticas.  Muestras faciles o alejadas de la frontera reciben
pocas o ninguna.

Ventaja sobre SMOTE: no genera ruido en zonas limpias de la frontera,
se concentra donde el clasificaria mal.

Uso directo:
    python src/balance/oversampling/s04_adasyn.py
        --input data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/adasyn/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL, ID_COL,
    ADASYN_K_NEIGHBORS,
    BAL_REPORTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADASYN Oversampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ADASYN: genera mas sinteticas donde la clase minoritaria
    esta mas rodeada por la mayoria.
    """
    col_preservar = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col_preservar = [c for c in col_preservar
                     if c not in (TARGET_COL, ID_COL)]
    col_num = [c for c in df.columns
               if c not in col_preservar + [TARGET_COL, ID_COL]]

    X = df[col_num].copy()
    y = df[TARGET_COL].copy()

    # Imputamos los valores nulos (NaN) en columnas numéricas usando la mediana
    # para evitar que se descarten casi todos los datos
    for col in X.columns:
        if X[col].isna().any():
            mediana = X[col].median()
            if pd.isna(mediana):
                mediana = 0.0
            X[col] = X[col].fillna(mediana)

    X_cl = X
    y_cl = y

    print(f"    Filas validas para ADASYN: {len(X_cl):,} de {len(df):,} (NaNs imputados con mediana)")

    adasyn = ADASYN(
        n_neighbors = ADASYN_K_NEIGHBORS,
        random_state= RANDOM_STATE,
    )
    X_res, y_res = adasyn.fit_resample(X_cl, y_cl)

    df_res          = pd.DataFrame(X_res, columns=col_num)
    df_res[TARGET_COL] = y_res.values

    # Marcar sinteticas
    es_sintetica = np.zeros(len(X_res), dtype=bool)
    es_sintetica[len(y_cl):] = True
    df_res["es_sintetico_adasyn"] = es_sintetica

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s04] ADASYN Oversampling")
    print(f"  {'-' * 52}")
    print(f"  Entrada : {args.input}")

    df    = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    sinteticos = int(df_bal.get("es_sintetico_adasyn",
                                pd.Series(dtype=bool)).sum())
    print(f"  Muestras sinteticas generadas: {sinteticos:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"\n  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica":        "adasyn",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
