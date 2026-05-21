"""
s05_borderline_smote.py
=========================
Borderline-SMOTE: variante de SMOTE que solo genera sinteticas a partir
de las muestras de la clase minoritaria que estan en la frontera con la
mayoria (es decir, sus k vecinos mas cercanos son en su mayoria de la
clase mayoritaria).

Ventaja: evita generar sinteticas en zonas limpias de la clase minoritaria,
lo que reduce el ruido y mejora la frontera de decision.

Uso directo:
    python src/balance/oversampling/s05_borderline_smote.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/borderline_smote/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL, ID_COL,
    BORDERLINE_K_NEIGH, BORDERLINE_KIND,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Borderline-SMOTE Oversampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica Borderline-SMOTE sobre las muestras fronterizas de la clase minoritaria."""
    col_preservar = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col_preservar = [c for c in col_preservar
                     if c not in (TARGET_COL, ID_COL)]
    col_num = [c for c in df.columns
               if c not in col_preservar + [TARGET_COL, ID_COL]]

    X = df[col_num].copy()
    y = df[TARGET_COL].copy()

    mask = X.notna().all(axis=1)
    X_cl = X[mask]
    y_cl = y[mask]

    print(f"    Filas validas para Borderline-SMOTE: {len(X_cl):,} de {len(df):,}")

    bl_smote = BorderlineSMOTE(
        k_neighbors   = BORDERLINE_K_NEIGH,
        m_neighbors   = BORDERLINE_K_NEIGH + 2,
        sampling_strategy = "auto",
        random_state  = RANDOM_STATE,
    )
    X_res, y_res = bl_smote.fit_resample(X_cl, y_cl)

    df_res         = pd.DataFrame(X_res, columns=col_num)
    df_res[TARGET_COL] = y_res.values

    indices_res = bl_smote.sample_indices_
    es_sintetica = np.zeros(len(X_res), dtype=bool)
    es_sintetica[indices_res] = True
    es_sintetica[:len(y_cl)] = False
    df_res["es_sintetico_borderline"] = es_sintetica

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s05] Borderline-SMOTE Oversampling")
    print(f"  {'─' * 52}")
    print(f"  Entrada : {args.input}")

    df    = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    sinteticos = int(df_bal.get("es_sintetico_borderline",
                                pd.Series(dtype=bool)).sum())
    print(f"  Muestras sinteticas generadas: {sinteticos:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"\n  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica":        "borderline_smote",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
