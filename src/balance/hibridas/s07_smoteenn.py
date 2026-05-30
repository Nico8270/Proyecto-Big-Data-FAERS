"""
s04_smoteenn.py
================
SMOTE + ENN: combina generacion de muestras sinteticas (SMOTE)
con eliminacion de muestras ruidosas (Edit Nearest Neighbours).

Flujo:
  1. SMOTE genera sinteticas de las clases minoritarias.
  2. ENN elimina cualquier muestra (en TODAS las clases) cuyos
     k vecinos mas cercanos no comparten su etiqueta.

Uso directo:
    python src/balance/s04_smoteenn.py

Uso desde balance_main.py:
    python src/balance/s04_smoteenn.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/smoteenn/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL,
    SMOTE_K_NEIGHBORS, SMOTE_SAMPLING,
    ENN_N_NEIGHBORS,   ENN_KIND_SEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMOTE + ENN Combined Sampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica SMOTEENN en dos pasos explicitos:
      1. SMOTE -> oversampling de minoritarias
      2. ENN  -> limpieza de muestras ruidosas
    """
    # Columnas no-numericas
    columnas_no_numericas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    columnas_id = [c for c in [TARGET_COL, "primaryid"]
                   if c in df.columns]
    columnas_preservar = [c for c in columnas_no_numericas
                          if c not in columnas_id]

    columnas_numericas = [c for c in df.columns
                          if c not in columnas_preservar
                          and c != "primaryid"]

    X = df[columnas_numericas].copy()
    y = df[TARGET_COL].copy()

    # Imputamos los valores nulos (NaN) en columnas numéricas usando la mediana
    # para evitar que se descarten casi todos los datos (el descarte drástico dejaba solo 45 filas válidas).
    for col in X.columns:
        if X[col].isna().any():
            mediana = X[col].median()
            if pd.isna(mediana):
                mediana = 0.0
            X[col] = X[col].fillna(mediana)

    X_cl  = X
    y_cl  = y

    print(f"    Filas validas: {len(X_cl):,} de {len(df):,} (NaNs imputados con mediana)")

    # Paso 1 — SMOTE
    smote = SMOTE(
        sampling_strategy = SMOTE_SAMPLING,
        k_neighbors        = SMOTE_K_NEIGHBORS,
        random_state       = RANDOM_STATE,
    )
    X_sm, y_sm = smote.fit_resample(X_cl, y_cl)
    print(f"    Tras SMOTE: {len(X_sm):,} filas")

    # Paso 2 — ENN (EditedNearestNeighbours es determinista, no recibe random_state)
    enn = EditedNearestNeighbours(
        n_neighbors   = ENN_N_NEIGHBORS,
        kind_sel      = ENN_KIND_SEL,
    )
    X_res, y_res = enn.fit_resample(X_sm, y_sm)
    print(f"    Tras ENN:   {len(X_res):,} filas  "
          f"(eliminadas {len(X_sm) - len(X_res):,})")

    # Reconstruir
    df_res        = pd.DataFrame(X_res, columns=columnas_numericas)
    df_res[TARGET_COL] = y_res.values
    df_res        = df_res.reset_index(drop=True)

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s04] SMOTE + ENN")
    print("  " + "-" * 52)
    print(f"  Entrada : {args.input}")

    df            = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"\n  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica": "smoteenn",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
