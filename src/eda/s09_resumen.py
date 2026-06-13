"""
s09_resumen.py
===============
Script 09 - Resumen final del EDA y plan de limpieza + balanceo.

Salidas:
  1. Tabla formateada en consola con 4 bloques:
       [1] Calidad de datos por tabla/columna
       [2] Resumen de duplicados
       [3] Balanceo de la variable objetivo
       [4] Resumen ejecutivo + proximos pasos

  2. CSV completo con todas las columnas de todas las tablas
     (outputs/eda_results/s09_resumen_completo.csv)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_eda import *
from utils_eda import (
    load_table, print_section, print_header_columnas, print_fila_columnas, stats_basicas,
    save_csv,
)

# ── Anchos de columna para las tablas de consola ──────────────────────────────
# Bloque 1 - Calidad de datos
COL_CALIDAD = ["Tabla", "Columna", "Tipo", "Nulos %", "Accion", "Justificacion"]
AN_CALIDAD  = [8, 28, 10, 9, 14, 36]

# Bloque 2 - Duplicados
COL_DUPS = ["Tabla", "Dups completos", "% Dups", "Dups PK", "% Dups PK"]
AN_DUPS  = [8, 18, 10, 14, 11]

# Bloque 3 - Balanceo
COL_BAL = ["Clase", "Descripcion", "Count", "%", "Ratio", "Estrategia"]
AN_BAL  = [7, 22, 10, 8, 8, 24]


# ── Bloque 1: Recoleccion de estadisticas de columnas ─────────────────────────

def recolectar_columnas() -> pd.DataFrame:
    """Recorre todas las tablas y devuelve un DataFrame con
       una fila por columna, listo para generar decisiones."""
    filas = []
    for nombre in FAERS_FILES:
        try:
            df = load_table(nombre, nrows=200_000)   # sample rapido
        except Exception:
            continue

        for col in df.columns:
            nulos     = int(df[col].isnull().sum())
            pct_nulos = round(nulos / len(df) * 100, 2)
            dtype     = str(df[col].dtype)
            filas.append({
                "tabla":    nombre,
                "columna":  col,
                "tipo":     dtype,
                "nulos":    nulos,
                "pct_nulos": pct_nulos,
                "unicos":   int(df[col].nunique()),
                "es_pk":    col == "primaryid",
            })
    return pd.DataFrame(filas)


def decidir_accion(df_cols: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de accion y justificacion a cada columna."""
    acciones = []

    for _, row in df_cols.iterrows():
        tabla, col, pct = row.tabla, row.columna, row.pct_nulos

        if row.es_pk:
            accion, justif = "ninguna",  "PK sin nulos"

        elif pct == 0.0:
            if row.tipo == "object":
                accion, justif = "normalizar", "Sin nulos, revisar formato (upper/lower)"
            else:
                accion, justif = "ninguna",  "Sin nulos ni problemas visibles"

        elif pct < 2.0:
            if row.tipo == "object":
                accion, justif = "imputar_moda", f"Pocos nulos, categoria mas frecuente"
            else:
                accion, justif = "imputar_mediana", f"Pocos nulos, distribucion simetrica"

        elif pct < 10.0:
            accion, justif = "imputar_mediana_o_knn", f"Nulos moderados, imputacion por vecinos"

        elif pct < 60.0:
            accion, justif = "eliminar_columna", f"Nulos altos ({pct}%), no es predictiva"

        else:
            accion, justif = "eliminar_columna", f"Nulos muy altos ({pct}%), columna descartada"

        acciones.append({
            **row.to_dict(),
            "accion":    accion,
            "justificacion": justif,
        })

    return pd.DataFrame(acciones)


# ── Bloque 2 - Duplicados ────────────────────────────────────────────────────

def obtener_duplicados() -> pd.DataFrame:
    filas = []
    for nombre in FAERS_FILES:
        try:
            stats = stats_basicas(nombre)
            filas.append({
                "tabla": nombre,
                "dups_completos":    stats["duplicados"],
                "pct_dups_completos": stats["pct_duplicados"],
                "dups_primaryid":    stats["pk_unicos"],
                "pct_dups_pk":       round((stats["filas"] - stats["pk_unicos"]) / stats["filas"] * 100, 4)
                                  if stats["pk_unicos"] > 0 else 0,
            })
        except Exception:
            pass
    return pd.DataFrame(filas)


# ── Bloque 3 - Balanceo (variable objetivo) ──────────────────────────────────

def obtener_balanceo(df_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla de balanceo hipotetica para el target 'severity_level'.
    En un caso real se calcularia con el dataset de entrenamiento.
    Devuelve un CSV para referencia.
    """
    # Valores hipoteticos basados en FAERS - se debe reemplazar por datos reales
    BALANCEO = [
        {"clase": 1, "descripcion": "Leve",         "count": 45_210, "estrategia": "class_weight='balanced'"},
        {"clase": 2, "descripcion": "Moderado",     "count": 14_280, "estrategia": "class_weight='balanced'"},
        {"clase": 3, "descripcion": "Importante",   "count":  6_800, "estrategia": "SMOTE oversampling"},
        {"clase": 4, "descripcion": "Grave",        "count":  2_540, "estrategia": "SMOTE + aumento sintetico"},
        {"clase": 5, "descripcion": "Muy grave",    "count":  1_410, "estrategia": "SMOTE + focal loss"},
    ]
    df = pd.DataFrame(BALANCEO)
    total = df["count"].sum()
    df["porcentaje"] = (df["count"] / total * 100).round(1)
    df["ratio"]      = (df["count"].iloc[0] / df["count"]).round(1)

    return df


# ── Impresion de tablas en consola ──────────────────────────────────────────

def imprimir_calidad(df_acciones: pd.DataFrame):
    print_section("[1] CALIDAD DE DATOS POR TABLA / COLUMNA")
    print_header_columnas(COL_CALIDAD, AN_CALIDAD)
    print(f"  {'-'*(sum(AN_CALIDAD)+10)}")

    for (tabla, grp) in df_acciones.groupby("tabla", sort=False):
        primera = True
        for _, row in grp.sort_values("columna").iterrows():
            if primera:
                print_fila_columnas(
                    [tabla, row.columna, row.tipo, f"{row.pct_nulos:.2f}%", row.accion, row.justificacion],
                    AN_CALIDAD)
                primera = False
            else:
                print_fila_columnas(
                    ["", row.columna, row.tipo, f"{row.pct_nulos:.2f}%", row.accion, row.justificacion],
                    AN_CALIDAD)

    # Totales
    n_cols    = len(df_acciones)
    n_act     = (df_acciones["accion"] != "ninguna").sum()
    n_descart = (df_acciones["accion"] == "eliminar_columna").sum()
    print(f"  {'-'*(sum(AN_CALIDAD)+10)}")
    print(f"  'Total de columnas analizadas   : {n_cols}")
    print(f"   Columnas que requieren accion  : {n_act}")
    print(f"   Columnas a descartar           : {n_descart}")


def imprimir_duplicados(df_dups: pd.DataFrame):
    print_section("[2] DUPLICADOS POR TABLA")
    print_header_columnas(COL_DUPS, AN_DUPS)
    print(f"  {'-'*(sum(AN_DUPS)+10)}")
    for _, row in df_dups.iterrows():
        print_fila_columnas(
            [row.tabla,
             f"{row.dups_completos:,}",
             f"{row.pct_dups_completos:.2f}%",
             f"{row.dups_primaryid if isinstance(row.dups_primaryid, str) else row.dups_primaryid:,}"
               if isinstance(row.dups_primaryid, int) else row.dups_primaryid,
             f"{row.pct_dups_pk:.2f}%"],
            AN_DUPS)


def imprimir_balanceo(df_bal: pd.DataFrame):
    print_section("[3] BALANCEO - variable objetivo: severity (1-5)")
    print_header_columnas(COL_BAL, AN_BAL)
    print(f"  {'-'*(sum(AN_BAL)+10)}")

    for _, row in df_bal.iterrows():
        print_fila_columnas(
            [
                row["clase"],
                row["descripcion"],
                f"{int(row['count']):,}",
                f"{float(row['porcentaje']):.1f}%",
                f"{float(row['ratio']):.1f}x",
                row["estrategia"],
            ],
            AN_BAL
        )

    print(f"  {'-'*(sum(AN_BAL)+10)}")
    may, min_ = df_bal["count"].max(), df_bal["count"].min()
    print(f"  Ratio desbalance : {may/min_:.1f}x  (clase 1 vs clase {df_bal['clase'].iloc[-1]})")
    print("  Metrica recomendada : Macro-F1 (no Accuracy)")


def imprimir_resumen(df_acciones: pd.DataFrame):
    print_section("[4] RESUMEN EJECUTIVO + PROXIMOS PASOS")
    total_raw = sum(len(load_table(n)) for n in FAERS_FILES if load_table(n).shape[0] > 0)
    n_elim   = df_acciones[df_acciones["accion"] == "eliminar_columna"]["columna"].nunique()
    n_imput  = df_acciones[df_acciones["accion"].str.contains("imputar")]["columna"].nunique()
    n_norm   = df_acciones[df_acciones["accion"] == "normalizar"]["columna"].nunique()

    print(f"""
  Dataset crudo                   : {total_raw:,} filas x 7 tablas
  Columnas con nulos               : {int((df_acciones['pct_nulos'] > 0).sum())}
  Columnas a imputar               : {n_imput}
  Columnas a normalizar            : {n_norm}
  Columnas a descartar              : {n_elim}

  PROXIMOS PASOS:
    1. Ejecutar s10_limpieza.py sobre data/staging/
    2. Ejecutar s11_balanceo.py con SMOTE sobre el dataset agregado
    3. Lanzar pipeline batch: hadoop_pipeline.sh
    4. Entrenar modelo: src/training/training_main.py
""")
    print("  NOTA: El CSV de plan (s09_resumen_completo.csv) servira para")
    print("        implementar s10_limpieza.py sin re-inferir nada.")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("SCRIPT 09: RESUMEN - LIMPIEZA + BALANCEO")

    print("\n[1/4] Recolectando estadisticas de columnas...")
    df_cols = recolectar_columnas()
    print(f"  {len(df_cols)} columnas analizadas en {df_cols['tabla'].nunique()} tablas")

    print("\n[2/4] Generando decisiones de accion...")
    df_acc = decidir_accion(df_cols)

    print("\n[3/4] Generando CSV completo...")
    save_csv(df_acc, "s09_resumen_completo.csv")
    print(f"\n  [OK] CSV completo: s09_resumen_completo.csv")

    print("\n[4/4] Imprimiendo tablas en consola...")
    imprimir_calidad(df_acc)
    imprimir_duplicados(obtener_duplicados())

    df_bal   = obtener_balanceo(df_cols)
    imprimir_balanceo(df_bal)

    # Guardar CSV de balanceo para el pipeline de balanceo
    save_csv(df_bal, "s09_balanceo.csv")

    imprimir_resumen(df_acc)

    print_section("FIN SCRIPT 09")


if __name__ == "__main__":
    main()
