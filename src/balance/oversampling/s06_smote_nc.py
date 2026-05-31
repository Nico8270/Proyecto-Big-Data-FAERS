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
    """
    Aplica SMOTENC: SMOTE con soporte de columnas categoricas de baja cardinalidad.
    Excluye columnas de alta cardinalidad para evitar desbordamientos de memoria.
    """
    # Columnas no-numericas totales
    col_no_num_total = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col_no_num_total = [c for c in col_no_num_total if c not in (TARGET_COL, ID_COL)]

    # Separar en baja y alta cardinalidad (umbral = 100)
    col_preservar = []
    col_alta_card = []
    for c in col_no_num_total:
        card = df[c].nunique()
        if card <= 100:
            col_preservar.append(c)
        else:
            col_alta_card.append((c, card))

    # Explicación en consola súper amigable
    print(f"  ==============================================================")
    print(f"  [RAM OPTIMIZATION] Optimizacion Inteligente de RAM para SMOTE-NC")
    print(f"  ==============================================================")
    print(f"  Para evitar que tu computadora se quede sin memoria RAM (Out of Memory)")
    print(f"  e impedir distorsiones estadisticas, hemos analizado la cardinalidad:")
    print(f"  - Columnas categoricas aptas (<= 100 clases): {len(col_preservar)}")
    if col_alta_card:
        print(f"  - Excluidas temporalmente por alta cardinalidad (> 100 clases): {len(col_alta_card)}")
        for c, card in col_alta_card:
            print(f"    * {c:<30} | {card:,} clases unicas")
        print(f"\n  [PROCESO] Estas columnas excluidas se preservaran intactas para los")
        print(f"  datos originales y se rellenaran con su moda/valor mas comun")
        print(f"  para las nuevas filas sinteticas creadas por SMOTE-NC.")
    print(f"  --------------------------------------------------------------\n")

    col_num = [c for c in df.columns
               if c not in col_no_num_total and c not in (TARGET_COL, ID_COL)]

    # Orden de features: primero numericas, luego categoricas de baja cardinalidad
    columnas_orden = col_num + col_preservar
    X = df[columnas_orden].copy()
    y = df[TARGET_COL].copy()

    # Convertir categoricas a string para SMOTENC
    for c in col_preservar:
        X[c] = X[c].astype(str)

    # Indices de las columnas categoricas en la matriz ordenada
    cat_indices = [columnas_orden.index(c) for c in col_preservar]

    # Imputamos los valores nulos (NaN) en columnas numéricas usando la mediana
    # para evitar que se descarten casi todos los datos
    for col in col_num:
        if X[col].isna().any():
            mediana = X[col].median()
            if pd.isna(mediana):
                mediana = 0.0
            X[col] = X[col].fillna(mediana)

    X_cl = X
    y_cl = y

    print(f"    Filas validas para SMOTE-NC: {len(X_cl):,} de {len(df):,} (NaNs imputados con mediana)")
    print(f"    Columnas categoricas en el calculo: {len(cat_indices)}  "
          f"({len(cat_indices)/max(len(columnas_orden),1)*100:.0f}%)")

    smotenc = SMOTENC(
        categorical_features = cat_indices,
        k_neighbors          = SMOTE_K_NEIGHBORS,
        sampling_strategy    = SMOTE_SAMPLING,
        random_state         = RANDOM_STATE,
    )
    X_res, y_res = smotenc.fit_resample(X_cl, y_cl)

    # Reconstruir DataFrame con columnas procesadas
    df_res         = pd.DataFrame(X_res, columns=columnas_orden)
    df_res[TARGET_COL] = y_res.values

    # Reincorporar las columnas de alta cardinalidad
    if col_alta_card:
        for c, _ in col_alta_card:
            valores = np.empty(len(df_res), dtype=object)
            # Copiar originales
            valores[:len(y_cl)] = df[c].values
            # Rellenar sinteticas con la moda
            moda = df[c].mode()[0] if not df[c].mode().empty else "SINTETICO"
            valores[len(y_cl):] = f"{moda}"
            df_res[c] = valores

    # Reincorporar primaryid si existe en el df original
    if ID_COL in df.columns:
        valores_id = np.empty(len(df_res), dtype=object)
        valores_id[:len(y_cl)] = df[ID_COL].values
        valores_id[len(y_cl):] = "SINTETICO"
        df_res[ID_COL] = valores_id

    es_sintetica = np.zeros(len(X_res), dtype=bool)
    es_sintetica[len(y_cl):] = True
    df_res["es_sintetico_smotenc"] = es_sintetica

    return df_res


def main():
    args = parse_args()
    t0   = time.time()

    print(f"\n  [s06] SMOTE-NC Oversampling")
    print(f"  {'-' * 52}")
    print(f"  Entrada : {args.input}")

    df    = pd.read_parquet(args.input)
    conteos_antes = df[TARGET_COL].value_counts().to_dict()
    print(f"  Registros originales: {len(df):,}")

    try:
        df_bal = aplicar(df)
    except MemoryError:
        print(f"\n  [ERROR DE MEMORIA] No se pudo ejecutar SMOTE-NC.")
        print(f"  Detalle: La alta cardinalidad de las columnas categoricas")
        print(f"  y el tamaño del dataset causaron que Numpy se quedara sin memoria RAM.")
        print(f"  Se recomienda usar SMOTE estandar o SMOTE+ENN en su lugar.")
        sys.exit(1)

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
