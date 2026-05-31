"""
s03_smote.py
=============
SMOTE (Synthetic Minority Over-sampling Technique): genera muestras
sinteticas de la clase minoritaria mediante interpolacion entre
vecinos mas cercanos, sin duplicar registros existentes.

Uso directo:
    python src/balance/s03_smote.py

Uso desde balance_main.py:
    python src/balance/s03_smote.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/smote/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL, SMOTE_K_NEIGHBORS, SMOTE_SAMPLING,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMOTE Oversampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica SMOTE sobre la clase minoritaria.

    Maneja columnas no-numericas excluyendolas del resampling;
    las reincorpora sin modificar despues del fit_resample.
    """
    # Separar ID y target
    id_col = "primaryid" if "primaryid" in df.columns else None

    # Columnas no-numericas que se preservan tal cual
    columnas_no_numericas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    columnas_id = [c for c in [TARGET_COL, "primaryid"] if c in df.columns]

    columnas_preservar = [c for c in columnas_no_numericas if c not in columnas_id]
    columnas_numericas = [c for c in df.columns if c not in columnas_preservar and c != "primaryid"]

    # Construir matriz solo con features para SMOTE
    X_num = df[columnas_numericas].copy()
    y     = df[TARGET_COL].copy()

    # Imputamos los valores nulos (NaN) en columnas numéricas usando la mediana
    # para evitar que se descarten casi todos los datos
    for col in X_num.columns:
        if X_num[col].isna().any():
            mediana = X_num[col].median()
            if pd.isna(mediana):
                mediana = 0.0
            X_num[col] = X_num[col].fillna(mediana)

    X_num_clean  = X_num
    y_clean      = y

    print(f"    Filas validas para SMOTE: {len(X_num_clean):,} de {len(df):,} (NaNs imputados con mediana)")

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING,
        k_neighbors      =SMOTE_K_NEIGHBORS,
        random_state     =RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_num_clean, y_clean)

    # Reconstruir DataFrame
    df_res          = pd.DataFrame(X_res, columns=columnas_numericas)
    df_res[TARGET_COL] = y_res.values

    # Muestras sinteticas generadas solo para las clases minoritarias
    es_sintetica = np.zeros(len(X_res), dtype=bool)
    es_sintetica[len(y_clean):] = True          # las generadas despues de la longitud de original son sinteticas
    df_res["es_sintetico_smote"] = es_sintetica

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s03] SMOTE Oversampling")
    print("  " + "-" * 52)
    print(f"  Entrada : {args.input}")

    df    = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    sinteticos = int(df_bal.get("es_sintetico_smote", pd.Series(dtype=bool)).sum())
    print(f"  Muestras sinteticas generadas: {sinteticos:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"\n  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica": "smote",
        "filas_originales":  int(len(df)),
        "filas_resultido":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
