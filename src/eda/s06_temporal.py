"""
s06_temporal.py
================
Script 06 — Analisis temporal de la tabla DEMO.

Salidas:
  • Serie mensual de reportes (linea)
  • Heatmap de reportes por mes y pais (top 10 paises)
  • CSV con serie mensual
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
    print_section("SCRIPT 06: ANALISIS TEMPORAL")

    print("\n[1/3] Cargando tabla DEMO...")
    demo = load_table("DEMO", usecols=["primaryid", "rept_dt", "fda_dt", "reporter_country"])
    print(f"  {len(demo):,} filas")

    # ── Serie mensual de reportes ─────────────────────────────────────────
    print("\n[2/3] Serie mensual de reportes...")
    demo["fda_dt_parsed"] = pd.to_datetime(
        demo["fda_dt"].astype(str).str[:6], format="%Y%m", errors="coerce"
    )
    serie_mensual = (
        demo.dropna(subset=["fda_dt_parsed"])
        .groupby(demo["fda_dt_parsed"].dt.to_period("M"))
        .size()
        .rename("reportes")
        .reset_index()
    )
    serie_mensual["mes"] = serie_mensual["fda_dt_parsed"].astype(str)

    save_csv(serie_mensual[["mes", "reportes"]], "s06_serie_mensual_reportes.csv")

    if len(serie_mensual) > 1:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(serie_mensual["mes"], serie_mensual["reportes"],
                color="teal", linewidth=1.5, marker="o", markersize=4)
        ax.fill_between(range(len(serie_mensual)), serie_mensual["reportes"],
                        alpha=0.15, color="teal")
        ax.set_title("Evolucion Mensual de Reportes FAERS",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Numero de Reportes")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_plot(fig, "s06_serie_mensual.png")

    # ── Heatmap pais × mes ─────────────────────────────────────────────────
    print("\n[3/3] Heatmap pais x mes...")
    top_paises = demo["reporter_country"].value_counts().head(10).index
    demo_top = demo[demo["reporter_country"].isin(top_paises)].copy()
    demo_top["mes"] = demo_top["fda_dt_parsed"].dt.to_period("M")

    pivot = (
        demo_top.groupby(["reporter_country", "mes"]).size()
        .reset_index(name="reportes")
        .pivot(index="reporter_country", columns="mes", values="reportes")
        .fillna(0)
    )
    if not pivot.empty and pivot.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot, cmap=PALETTE_MAIN, linewidths=0.3, ax=ax)
        ax.set_title("Reportes por Pais y Mes (Top 10 paises)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Pais (codigo ISO)")
        plt.tight_layout()
        save_plot(fig, "s06_heatmap_pais_mes.png")

    print_section("FIN SCRIPT 06")


if __name__ == "__main__":
    main()
