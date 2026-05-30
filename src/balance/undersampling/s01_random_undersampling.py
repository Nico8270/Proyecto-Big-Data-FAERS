"""
s01_undersampling_random.py
=============================
Random Undersampling: reduce la clase mayoritaria eliminando muestras
al azar hasta que su tamanio iguale al de la clase minoritaria.

Uso directo:
    python src/balance/s01_undersampling_random.py

Uso desde balance_main.py:
    python src/balance/s01_undersampling_random.py
        --input  data/clean_data/dataset_consolidado.parquet
        --output data/balanced_data/random_undersampling/dataset.parquet

Salida por stdout (ultima linea JSON):
    {"tecnica": "random_undersampling", "filas_originales": N, "filas_resultado": M, "tiempo_s": T}
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    RANDOM_STATE, TARGET_COL,
    BAL_REPORTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Undersampling")
    parser.add_argument("--input",  type=str, required=True, help="Ruta al dataset de entrada (.parquet)")
    parser.add_argument("--output", type=str, required=True, help="Ruta al dataset de salida (.parquet)")
    return parser.parse_args()


def aplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Random Undersampling: reduce la clase mayoritaria hasta igualar
    a la minoritaria.
    """
    conteos    = df[TARGET_COL].value_counts()
    clase_min  = conteos.idxmin()
    n_min      = conteos.min()

    df_min     = df[df[TARGET_COL] == clase_min]
    df_may     = df[df[TARGET_COL] != clase_min]

    rng  = np.random.default_rng(RANDOM_STATE)
    idx  = rng.choice(df_may.index, size=n_min, replace=False)
    df_may_down = df_may.loc[idx]

    df_bal = pd.concat([df_min, df_may_down]).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return df_bal


def main():
    args   = parse_args()
    t0     = time.time()

    print(f"\n  [s01] Random Undersampling")
    print(f"  {'─' * 52}")
    print(f"  Entrada : {args.input}")

    df = pd.read_parquet(args.input)
    print(f"  Registros originales: {len(df):,}")

    conteos_antes = df[TARGET_COL].value_counts().to_dict()

    df_bal = aplicar(df)

    conteos_despues = df_bal[TARGET_COL].value_counts().to_dict()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_parquet(args.output, index=False, compression="snappy")

    duracion = round(time.time() - t0, 1)

    print(f"  Registros resultantes: {len(df_bal):,}")
    print(f"  Tiempo: {duracion}s")

    # Resumen de distribucion
    print(f"\n  Distribucion ANTES:")
    for k in sorted(conteos_antes):
        print(f"    Clase {k}: {conteos_antes[k]:>8,}  "
              f"({conteos_antes[k]/len(df)*100:.1f}%)")

    print(f"\n  Distribucion DESPUES:")
    for k in sorted(conteos_despues):
        print(f"    Clase {k}: {conteos_despues[k]:>8,}  "
              f"({conteos_despues[k]/len(df_bal)*100:.1f}%)")

    resultado = {
        "tecnica": "random_undersampling",
        "filas_originales":  int(len(df)),
        "filas_resultado":   int(len(df_bal)),
        "tiempo_s":          duracion,
    }
    print(f"\n  {json.dumps(resultado)}")


if __name__ == "__main__":
    main()
