"""
s02_undersampling_tomek.py
=============================
Tomek Links undersampling: elimina pares de muestras (clase mayoria,
clase minoria) que son vecinos mas cercanos entre si y pertenecen
a clases distintas.  Esto limpia la frontera de decision.

Uso directo:
    python src/balance/s02_undersampling_tomek.py

Uso desde balance_main.py:
    python src/balance/s02_undersampling_tomek.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/tomek/dataset.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import RANDOM_STATE, TARGET_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tomek Links Undersampling")
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Tomek Links: elimina pares mayoria-minoria que son vecinos
    mas cercanos, reduciendo el solapamiento en la frontera.
    """
    # Columnas no-numericas
    columnas_no_numericas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    columnas_id = [c for c in [TARGET_COL, "primaryid"] if c in df.columns]
    columnas_preservar = [c for c in columnas_no_numericas if c not in columnas_id]
    columnas_numericas = [c for c in df.columns if c not in columnas_preservar and c != "primaryid" and c != TARGET_COL]

    X_num = df[columnas_numericas].copy()
    y = df[TARGET_COL].copy()

    # Imputar NaNs en columnas numericas usando la mediana
    for col in X_num.columns:
        if X_num[col].isna().any():
            mediana = X_num[col].median()
            if pd.isna(mediana):
                mediana = 0.0
            X_num[col] = X_num[col].fillna(mediana)

    tomek = TomekLinks(sampling_strategy="auto")
    tomek.fit_resample(X_num, y)

    # Obtener los indices que se deben conservar
    keep_indices = tomek.sample_indices_

    # Retornar el dataset original filtrado con los indices conservados
    df_res = df.iloc[keep_indices].reset_index(drop=True)
    return df_res


def main():
    args  = parse_args()
    t0    = time.time()

    print(f"\n  [s02] Tomek Links Undersampling")
    print(f"  {'-' * 52}")
    print(f"  Entrada : {args.input}")

    df   = pd.read_parquet(args.input)
    print(f"  Registros originales: {len(df):,}")

    conteos_antes = df[TARGET_COL].value_counts().to_dict()

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"  Registros resultantes: {len(df_bal):,}")
    eliminados = len(df) - len(df_bal)
    print(f"  Muestras eliminadas  : {eliminados:,}")
    print(f"  Tiempo: {duracion}s")

    resultado = {
        "tecnica": "tomek",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
