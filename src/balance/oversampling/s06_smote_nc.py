"""
s06_smote_nc.py
================
SMOTE-NC (SMOTE Nominal-Continuous): extension de SMOTE que trabaja
sobre datasets que tienen columnas categoricas nominales mezcladas con
numericas.

No interpola sobre columnas categoricas.  Procesa las categoricas con
distancia de Hamming y las numericas con distancia euclidiana, combinando
ambas en una matriz de vecinos consistente.

Uso directo:
    python src/balance/oversampling/s06_smote_nc.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/smote_nc/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL, ID_COL,
    SMOTE_K_NEIGHBORS, SMOTE_SAMPLING,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMOTE-NC Oversampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica SMOTENC: SMOTE con soporte de columnas categoricas."""
    col_preservar = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col_preservar = [c for c in col_preservar
                     if c not in (TARGET_COL, ID_COL)]
    col_num = [c for c in df.columns
               if c not in col_preservar + [TARGET_COL, ID_COL]]

    # Orden de features: primero numericas, luego categoricas
    columnas_orden = col_num + col_preservar
    X = df[columnas_orden].copy()
    y = df[TARGET_COL].copy()

    # Convertir categoricas a string para SMOTENC
    for c in col_preservar:
        X[c] = X[c].astype(str)

    # Indices de las columnas categoricas en la matriz ordenada
    cat_indices = [columnas_orden.index(c) for c in col_preservar]

    mask = X[col_num].notna().all(axis=1)
    X_cl = X[mask]
    y_cl = y[mask]

    print(f"    Filas validas para SMOTE-NC: {len(X_cl):,} de {len(df):,}")
    print(f"    Columnas categoricas: {len(cat_indices)}  "
          f"({len(cat_indices)/max(len(columnas_orden),1)*100:.0f}%)")

    smotenc = SMOTENC(
        categorical_features = cat_indices,
        k_neighbors          = SMOTE_K_NEIGHBORS,
        sampling_strategy    = SMOTE_SAMPLING,
        random_state         = RANDOM_STATE,
    )
    X_res, y_res = smotenc.fit_resample(X_cl, y_cl)

    df_res         = pd.DataFrame(X_res, columns=columnas_orden)
    df_res[TARGET_COL] = y_res.values

    indices_res = smotenc.sample_indices_
    es_sintetica = np.zeros(len(X_res), dtype=bool)
    es_sintetica[indices_res] = True
    es_sintetica[:len(y_cl)] = False
    df_res["es_sintetico_smotenc"] = es_sintetica

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s06] SMOTE-NC Oversampling")
    print(f"  {'─' * 52}")
    print(f"  Entrada : {args.input}")

    df    = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    sinteticos = int(df_bal.get("es_sintetico_smotenc",
                                pd.Series(dtype=bool)).sum())
    print(f"  Muestras sinteticas generadas: {sinteticos:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"\n  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica":        "smote_nc",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
