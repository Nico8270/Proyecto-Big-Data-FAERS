"""
s00_diagnostico.py
==================
Script de diagnostico — no modifica datos.

Lee el dataset consolidado limpio y las salidas del EDA para calcular
todos los indicadores que necesitan las reglas de seleccion de tecnica.

Entrada:
  - data/clean_data/dataset_consolidado.parquet
  - outputs/eda_results/s09_resumen_completo.csv
  - outputs/eda_results/s09_balanceo.csv

Salida:
  - Imprime en consola el diagnostico formateado
  - Guarda cache en src/balance/.cache_diagnostico.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from balance.config_balance import (
    CONSOLIDADO, PLAN_CSV, BALANCEO_CSV, TARGET_COL, CACHE_DIAGNOSTICO, TARGET_VALUES,
)
from balance.rules_balance import PCT_CATEGORICAS_LIM, CLASE_MUY_MIN_PCT, CARDINALIDAD_LIM


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carga el consolidado y el plan EDA si existe."""
    df = pd.read_parquet(CONSOLIDADO)
    plan = pd.read_csv(PLAN_CSV) if PLAN_CSV.exists() else None
    return df, plan


def medir_tamano(df: pd.DataFrame) -> dict:
    return {
        "n_filas":    int(len(df)),
        "n_columnas": int(df.shape[1]),
    }


def medir_desbalance(df: pd.DataFrame) -> dict:
    """Calcula ratio y conteos por clase."""
    conteos = df[TARGET_COL].value_counts().sort_index()
    if conteos.empty:
        return {"ratio": 0.0, "clase_mayoritaria": 0, "clase_minoritaria": 0}

    # Solo clases validas
    conteos = conteos[conteos.index.isin(TARGET_VALUES)]
    clase_mayor = int(conteos.idxmax())
    clase_menor = int(conteos.idxmin())
    ratio = float(conteos.max() / max(conteos.min(), 1))
    total = int(conteos.sum())

    clase_min_pct = float(conteos.min() / total) if total > 0 else 0.0

    return {
        "ratio":             round(ratio, 3),
        "clase_mayoritaria": clase_mayor,
        "clase_minoritaria": clase_menor,
        "conteos":           {int(k): int(v) for k, v in conteos.items()},
        "clase_min_pct":     round(clase_min_pct, 5),
        "filas_con_target":  total,
    }


def medir_features(df: pd.DataFrame, plan: pd.DataFrame | None) -> dict:
    """
    Mide la proporcion de columnas categoricas y la cardinalidad promedio.
    Si hay plan EDA lo usa para acelerar la deteccion de tipos.
    """
    num_col   = df.select_dtypes(include=["number"]).columns
    cat_col   = df.select_dtypes(include=["object", "category"]).columns

    total_cols  = len(df.columns)
    n_num       = len(num_col)
    n_cat       = len(cat_col)
    pct_cat     = round(n_cat / max(total_cols, 1), 4)

    # Cardinalidad promedio de columnas categoricas
    if n_cat > 0:
        card = [df[c].nunique() for c in cat_col]
        card_prom = round(sum(card) / len(card), 1)
        card_max  = int(max(card))
    else:
        card_prom = 0.0
        card_max  = 0

    return {
        "total_columnas":    total_cols,
        "columnas_numericas": n_num,
        "columnas_categoricas": n_cat,
        "pct_categoricas":   pct_cat,
        "cardinalidad_promedio": card_prom,
        "cardinalidad_maxima":   card_max,
    }


def cargar_plan_si_existe() -> dict | None:
    """Extrae metadatos del plan EDA si existe el CSV."""
    if PLAN_CSV.exists():
        try:
            df = pd.read_csv(PLAN_CSV)
            return {
                "existe":     True,
                "n_columnas": len(df),
                "n_eliminar": int((df.get("accion", pd.Series()) == "eliminar_columna").sum())
                                 if "accion" in df.columns else 0,
            }
        except Exception:
            return None
    return None


def guardar_cache(datos: dict):
    with open(CACHE_DIAGNOSTICO, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def imprimir(tam: dict, desb: dict, feat: dict, plan_info: dict | None) -> None:
    ancho = 60
    def linea(txt=""):
        if txt:
            print(f"  {txt}")
        else:
            print(f"  {'-' * ancho}")

    print(f"\n  DIAGNOSTICO DE DATOS")
    linea()

    print(f"  Tamano del dataset")
    print(f"    Filas       : {tam['n_filas']:>10,}")
    print(f"    Columnas    : {tam['n_columnas']:>10}")

    linea()
    print(f"  Desbalance (target = {TARGET_COL})")
    print(f"    Clase mayoritaria : {desb['clase_mayoritaria']}  "
          f"({desb['conteos'].get(desb['clase_mayoritaria'], 0):,} casos)")
    print(f"    Clase minoritaria : {desb['clase_minoritaria']}  "
          f"({desb['conteos'].get(desb['clase_minoritaria'], 0):,} casos)")
    print(f"    Ratio desbalance  : {desb['ratio']}x")
    pct = desb['clase_min_pct'] * 100
    frag = "MUY BAJO" if pct < 1.0 else "bajo" if pct < 5 else "normal"
    print(f"    Clase minoritaria : {pct:.2f}%  ({frag})")

    linea()
    print(f"  Caracteristicas de las features")
    print(f"    Columnas numericas   : {feat['columnas_numericas']}")
    print(f"    Columnas categoricas : {feat['columnas_categoricas']}  "
          f"({feat['pct_categoricas']*100:.0f}%)")
    print(f"    Cardinalidad prom.   : {feat['cardinalidad_promedio']:.0f}  "
          f"(max {feat['cardinalidad_maxima']})")

    if plan_info and plan_info.get("existe"):
        linea()
        print(f"  Plan EDA disponible")
        print(f"    Columnas analizadas   : {plan_info['n_columnas']}")
        print(f"    Columnas a descartar  : {plan_info['n_eliminar']}")

    linea()


def obtener_indices() -> dict:
    """Extrae los indices de regulacion en base al diagnostico."""
    tam  = medir_tamano(df)
    desb = medir_desbalance(df)
    feat = medir_features(df, plan)

    es_pequeno   = tam ["n_filas"]    < TAMANO_PEQ_MAX
    es_grande    = tam ["n_filas"]    >= TAMANO_GRANDE_MIN
    muchas_cat   = feat["pct_categoricas"] >= PCT_CATEGORICAS_LIM
    muy_pocos    = desb["clase_min_pct"]   < CLASE_MUY_MIN_PCT
    alta_card    = feat["cardinalidad_promedio"] > CARDINALIDAD_LIM
    ratio_alto   = desb["ratio"] >= RATIO_SMOTEENN
    ratio_medio  = RATIO_SMOTE <= desb["ratio"] < RATIO_SMOTEENN
    necesita_accion = desb["ratio"] > RATIO_SIN_ACCION

    return {
        "tam": tam, "desb": desb, "feat": feat,
        "es_pequeno": es_pequeno, "es_grande": es_grande,
        "muchas_cat": muchas_cat, "muy_pocos": muy_pocos,
        "alta_card": alta_card, "ratio_alto": ratio_alto,
        "ratio_medio": ratio_medio, "necesita_accion": necesita_accion,
    }


def construir_diagnostico_completo(df: pd.DataFrame,
                                    plan: pd.DataFrame | None) -> dict:
    """Compila todos los indicadores en un solo dict para el orquestador."""
    tam  = medir_tamano(df)
    desb = medir_desbalance(df)
    feat = medir_features(df, plan)
    plan_info = cargar_plan_si_existe()

    return {
        "tamano":      tam,
        "desbalance":  desb,
        "features":    feat,
        "plan_eda":    plan_info,
        "cache": {
            "n_filas":           tam["n_filas"],
            "ratio":             desb["ratio"],
            "pct_categoricas":   feat["pct_categoricas"],
            "clase_min_pct":     desb["clase_min_pct"],
            "cardinalidad_prom": feat["cardinalidad_promedio"],
            "clase_mayoritaria": desb["clase_mayoritaria"],
            "clase_minoritaria": desb["clase_minoritaria"],
        },
    }


# ── Punto de entrada ────────────────────────────────────────────────────────────

def main():
    global df, plan
    print("=" * 60)
    print("  DIAGNOSTICO DE DATOS — FAERS Q4 2025")
    print("=" * 60)

    print("\n  [PASO 0] Cargando dataset consolidado...")
    df   = pd.read_parquet(CONSOLIDADO)
    print(f"  Cargado: {CONSOLIDADO.name}  ({len(df):,} filas)")

    plan = pd.read_csv(PLAN_CSV) if PLAN_CSV.exists() else None
    if plan is not None:
        print(f"  [OK] Plan EDA: {PLAN_CSV.name}")

    diag = construir_diagnostico_completo(df, plan)

    imprimir(diag["tamano"], diag["desbalance"], diag["features"],
             diag["plan_eda"])

    guardar_cache(diag["cache"])
    print(f"  [OK] Cache de diagnostico guardado: {CACHE_DIAGNOSTICO.name}\n")
    print("=" * 60)
    print("  DIAGNOSTICO COMPLETADO")


if __name__ == "__main__":
    main()
