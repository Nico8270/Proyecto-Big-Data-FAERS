"""
s04_reacciones.py
==================
Script 04 — Analisis de reacciones adversas (tabla REAC).

Salidas:
  • Top 25 reacciones MedDRA PT
  • Distribucion de reacciones por caso (histograma)
  • CSV con top reacciones
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_eda import *
from utils_eda import load_table, save_plot, save_csv, print_section


def main():
    print_section("SCRIPT 04: REACCIONES ADVERSAS")

    print("\n[1/4] Cargando tabla REAC...")
    reac = load_table("REAC")
    print(f"  {len(reac):,} filas × {reac.shape[1]} columnas")

    # ── Top reacciones ────────────────────────────────────────────────────
    print("\n[2/4] Top 25 reacciones (MedDRA PT)...")
    if "pt" in reac.columns:
        top = (reac["pt"].str.strip().str.title().value_counts().head(25))
        print(f"\n  Top 5\n{top.head().to_string()}")

        fig, ax = plt.subplots(figsize=(12, 8))
        top.plot(kind="barh", ax=ax, color="darkorange", edgecolor="white")
        ax.set_title("Top 25 Reacciones Adversas Mas Reportadas (MedDRA PT)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Reportes")
        ax.invert_yaxis()
        plt.tight_layout()
        save_plot(fig, "s04_top25_reacciones.png")
        save_csv(top.rename("reportes").reset_index().rename(columns={"index": "reaccion_pt"}),
                 "s04_top25_reacciones.csv")

    # ── Reacciones por caso ───────────────────────────────────────────────
    print("\n[3/4] Reacciones por caso...")
    if "primaryid" in reac.columns:
        por_caso = reac.groupby("primaryid")["pt"].count()
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        clip = por_caso.clip(upper=20)
        ax.hist(clip, bins=20, color="orchid", edgecolor="white", alpha=0.85)
        ax.set_title("Numero de Reacciones Adversas por Caso (limitado a 20)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Cantidad de reacciones")
        ax.set_ylabel("Numero de casos")
        plt.tight_layout()
        save_plot(fig, "s04_reacciones_por_caso.png")

        stats = por_caso.describe().round(2).reset_index()
        stats.columns = ["estadistica", "valor"]
        save_csv(stats, "s04_estadisticas_por_caso.csv")

    # ── Resumen numerico ──────────────────────────────────────────────────
    print("\n[4/4] Resumen...")
    if "primaryid" in reac.columns and "pt" in reac.columns:
        casos_con_reaccion = reac["primaryid"].nunique()
        reacciones_unicas  = reac["pt"].nunique()
        print(f"  Reacciones totales       : {len(reac):,}")
        print(f"  Casos con >= 1 reaccion : {casos_con_reaccion:,}")
        print(f"  Reacciones distintas     : {reacciones_unicas:,}")

    print_section("FIN SCRIPT 04")


if __name__ == "__main__":
    main()
