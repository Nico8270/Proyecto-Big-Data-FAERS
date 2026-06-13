"""
src/hadoop/s04_safety_signals.py
==================================
CÁLCULO DE SEÑALES DE SEGURIDAD ESTADÍSTICAS (PRR Y CHI-CUADRADO)

Este script lee el dataset consolidado (generado por el Join de MapReduce),
calcula las métricas oficiales de farmacovigilancia (PRR y Chi-cuadrado con corrección de Yates)
para cada par Fármaco-Reacción, y guarda las alertas estadísticas en
outputs/mapreduce_results/top_safety_signals.csv.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "data" / "clean_data" / "dataset_consolidado.parquet"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "mapreduce_results"
OUTPUT_CSV    = OUTPUT_DIR / "top_safety_signals.csv"

def calcular_senales():
    print("\n" + "=" * 60)
    print("  CALCULADORA DE SEÑALES DE SEGURIDAD ESTADÍSTICAS (HADOOP / MAPREDUCE)")
    print("=" * 60)

    if not INPUT_PARQUET.exists():
        print(f"\n[ERROR] No se encontró el dataset consolidado en: {INPUT_PARQUET}")
        print("  Asegúrate de haber ejecutado la Opción 3 (Join con Hadoop y MapReduce).")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Cargando dataset consolidado: {INPUT_PARQUET.name}...")
    try:
        df = pd.read_parquet(INPUT_PARQUET)
        N = len(df)
        print(f"    -> total de reportes en base de datos (N) = {N:,}")

        # Normalizar nombres para evitar discrepancias
        df["drugname"] = df["drugname"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        df["pt"] = df["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

        print("  Calculando frecuencias marginales y co-ocurrencias...")
        
        # 1. Co-ocurrencias (a = count_a)
        cooc = df.groupby(["drugname", "pt"]).size().reset_index(name="count_a")
        
        # 2. Marginales por Fármaco (total_drug = a + b)
        drug_counts = df.groupby("drugname").size().reset_index(name="total_drug")
        
        # 3. Marginales por Reacción (total_reac = a + c)
        reac_counts = df.groupby("pt").size().reset_index(name="total_reac")

        # Unir tablas para tener todos los componentes en un DataFrame
        signals = cooc.merge(drug_counts, on="drugname", how="left")
        signals = signals.merge(reac_counts, on="pt", how="left")

        print("  Calculando métricas PRR y Chi-cuadrado para todos los pares...")

        # Fórmulas de farmacovigilancia:
        # a = count_a (fármaco + reacción)
        # b = total_drug - a (fármaco sin reacción)
        # c = total_reac - a (reacción sin fármaco)
        # d = N - (a + b + c) = N - total_drug - total_reac + a (ni fármaco ni reacción)
        
        a = signals["count_a"]
        b = signals["total_drug"] - a
        c = signals["total_reac"] - a
        d = N - (signals["total_drug"] + signals["total_reac"] - a)

        # Evitar división por cero sumando un valor infinitesimal (épsilon)
        eps = 1e-9

        # PRR = [a / (a + b)] / [c / (c + d)]
        tasa_exp = a / (a + b + eps)
        tasa_no_exp = c / (c + d + eps)
        signals["prr"] = tasa_exp / (tasa_no_exp + eps)

        # Chi-cuadrado con corrección de Yates para tablas 2x2
        # Chi2 = N * (|ad - bc| - N/2)^2 / [(a+b)(c+d)(a+c)(b+d)]
        numerador = N * (np.abs(a * d - b * c) - N / 2) ** 2
        denominador = (a + b) * (c + d) * (a + c) * (b + d)
        signals["chi2"] = numerador / (denominador + eps)

        # Filtrar señales significativas (estándar de farmacovigilancia):
        # - Al menos 3 casos reportados (count_a >= 3)
        # - PRR >= 2.0
        # - Chi-cuadrado >= 3.84 (P-value < 0.05)
        signals_filtradas = signals[
            (signals["count_a"] >= 3) & 
            (signals["prr"] >= 2.0) & 
            (signals["chi2"] >= 3.84)
        ].copy()

        # Ordenar por fuerza de la señal estadística (Chi-cuadrado descendente)
        signals_filtradas = signals_filtradas.sort_values(by="chi2", ascending=False)

        # Seleccionar y reordenar columnas finales para inference_helper.py
        columnas_finales = ["drugname", "pt", "count_a", "prr", "chi2"]
        df_csv = signals_filtradas[columnas_finales]

        print(f"  Guardando {len(df_csv):,} señales de seguridad detectadas...")
        df_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"  [OK] Señales guardadas exitosamente en: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    except Exception as e:
        print(f"\n[ERROR] Falló el cálculo de señales estadísticas: {e}")
        sys.exit(1)

    print("=" * 60)
    print("  PROCESO DE SEÑALES COMPLETADO CON ÉXITO")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    calcular_senales()
