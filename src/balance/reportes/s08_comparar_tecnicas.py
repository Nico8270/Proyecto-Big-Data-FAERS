"""
s05_comparar_tecnicas.py
=========================
Lee todos los datasets balanceados de data/balanced_data/ y genera
un CSV comparativo con metricas por tecnica.

Uso directo:
    python src/balance/s05_comparar_tecnicas.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from balance.config_balance import (
    BALANCED_DIR, BAL_REPORTS, TARGET_COL,
    ARCHIVO_SALIDA, ARCHIVO_METRICAS,
    RATIO_SIN_ACCION, TECNICAS,
)
from balance.rules_balance import ratio_aceptable


def recolectar_datasets() -> dict:
    """Recorre las subcarpetas de balanced_data y devuelve un dict {clave: df}."""
    datasets = {}
    for clave in TECNICAS:
        ruta = BALANCED_DIR / clave / ARCHIVO_SALIDA
        if ruta.exists():
            try:
                datasets[clave] = pd.read_parquet(ruta)
            except Exception:
                pass
    return datasets


def calcular_metricas(nombre: str, df: pd.DataFrame, df_original: pd.DataFrame) -> dict:
    """Calcula metricas de comparacion para un dataset balanceado."""
    conteos       = df[TARGET_COL].value_counts().sort_index()
    conteos_orig  = df_original[TARGET_COL].value_counts().sort_index()

    ratio = conteos.max() / conteos.min() if conteos.min() > 0 else float("inf")
    ratio_orig = conteos_orig.max() / conteos_orig.min() if conteos_orig.min() > 0 else float("inf")

    muestra_original = {c: conteos_orig.get(c, 0) for c in range(1, 6)}
    muestra_resultado = {c: conteos.get(c, 0) for c in range(1, 6)}

    return {
        "tecnica":           TECNICAS[nombre]["label"],
        "filas_originales":  int(len(df_original)),
        "filas_resultado":   int(len(df)),
        "ratio_original":    round(float(ratio_orig), 2),
        "ratio_resultado":   round(float(ratio), 2),
        "mejora_ratio_pct":  round((ratio_orig - ratio) / ratio_orig * 100, 1)
                             if ratio_orig > 0 else 0.0,
        "balance_ok":        ratio_aceptable(ratio),
        "clase1_original":   muestra_original.get(1, 0),
        "clase1_resultado":  muestra_resultado.get(1, 0),
        "clase5_original":   muestra_original.get(5, 0),
        "clase5_resultado":  muestra_resultado.get(5, 0),
    }


def imprimir_comparativo(metricas_lista: list):
    """Imprime tabla comparativa formateada en consola."""
    col = ["tecnica", "filas_resultado", "ratio_orig", "ratio_final", "mejora_pct", "OK"]
    an  = [30,              14,               12,         12,           12,          6]

    print("\n  COMPARATIVO DE TECNICAS DE BALANCEO")
    print(f"  {'─' * 88}")

    header = "  " + "  ".join(f"{c:<{a}}" for c, a in zip(col, an))
    print(header)
    print("  " + "  ".join("─" * (a - 2) for a in an))

    for m in metricas_lista:
        fila = [
            m["tecnica"],
            f"{m['filas_resultado']:,}",
            f"{m['ratio_original']}x",
            f"{m['ratio_resultado']}x",
            f"{m['mejora_ratio_pct']}%",
            "SI" if m["balance_ok"] else "NO",
        ]
        linea = "  " + "  ".join(f"{v:<{a}}" for v, a in zip(fila, an))
        print(linea)

    print(f"  {'─' * 88}")


def imprimir_distribucion_por_clase(datasets: dict, df_orig: pd.DataFrame):
    """Tabla de distribucion antes/despues por clase y tecnica."""
    print("\n  DISTRIBUCION POR CLASE (antes / despues)")
    print(f"  {'─' * 88}")

    header = f"  {'Clase':<8}" + "".join(
        f"  {TECNICAS[k]['label']:<22}" for k in datasets
    )
    print(header)

    orig_counts = df_orig[TARGET_COL].value_counts().sort_index()

    for clase in sorted(orig_counts.index):
        antes = f"{orig_counts.get(clase, 0):,}"
        despues_lista = []
        for clave, df in datasets.items():
            cnt = df[TARGET_COL].value_counts().get(clase, 0)
            despues_lista.append(f"{cnt:,}")
        linea = f"  {clase:<8}" + "  " + "  ".join(
            f"{d:<22}" for d in [antes] + despues_lista
        )
        print(linea)

    print(f"  {'─' * 88}")


def guardar_csv(metricas_lista: list):
    """Guarda el comparativo como CSV."""
    df_out = pd.DataFrame(metricas_lista)
    ARCHIVO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ARCHIVO_METRICAS, index=False, encoding="utf-8")
    print(f"\n  [OK] Comparativo guardado: {ARCHIVO_METRICAS}")


def main():
    print("=" * 60)
    print("  COMPARACION DE TECNICAS DE BALANCEO")
    print("=" * 60)

    # Dataset original
    from balance.config_balance import CONSOLIDADO
    if not CONSOLIDADO.exists():
        print(f"\n  [ERROR] No existe el consolidado: {CONSOLIDADO}")
        sys.exit(1)

    df_orig = pd.read_parquet(CONSOLIDADO)
    print(f"\n  Dataset original: {len(df_orig):,} filas × {df_orig.shape[1]} columnas")

    # Recolectar datasets balanceados
    datasets = recolectar_datasets()

    if not datasets:
        print("\n  [AVISO] No se encontraron datasets balanceados en "
              f"{BALANCED_DIR}.")
        print("  Ejecute primero alguna tecnica de balanceo.")
        sys.exit(0)

    print(f"\n  Tecnicas encontradas: {', '.join(TECNICAS[k]['label'] for k in datasets)}")

    # Calcular metricas
    metricas_lista = [calcular_metricas(k, v, df_orig) for k, v in datasets.items()]

    # Salidas
    imprimir_distribucion_por_clase(datasets, df_orig)
    imprimir_comparativo(metricas_lista)
    guardar_csv(metricas_lista)

    print("\n" + "=" * 60)
    print("  COMPARACION FINALIZADA")
    print("=" * 60)


if __name__ == "__main__":
    main()
