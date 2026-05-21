"""
s07_cruzado.py
===============
Script 07 — Analisis cruzado DRUG x REAC (JOIN por primaryid).

Salidas:
  • Heatmap top 10 farmacos sospechosos × top 10 reacciones
  • CSV con top 20 pares farmaco-reaccion
  • Estadisticas del JOIN
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
    print_section("SCRIPT 07: ANALISIS CRUZADO DRUG x REAC")

    print("\n[1/3] Cargando tablas DRUG y REAC...")
    drug = load_table("DRUG", usecols=["primaryid", "drugname", "role_cod"])
    reac = load_table("REAC", usecols=["primaryid", "pt"])
    print(f"  DRUG: {len(drug):,} filas  |  REAC: {len(reac):,} filas")

    # ── Filtrar solo farmaco sospechoso primario ──────────────────────────
    print("\n[2/3] Filtrando farmacos sospechosos primarios (PS)...")
    drug_ps = drug[drug["role_cod"] == "PS"].copy()
    drug_ps["drugname"] = drug_ps["drugname"].str.upper().str.strip()
    reac["pt"] = reac["pt"].str.strip().str.title()
    print(f"  Farmacos PS: {len(drug_ps):,}  |  Reacciones: {len(reac):,}")

    # ── JOIN ───────────────────────────────────────────────────────────────
    merged = pd.merge(drug_ps[["primaryid", "drugname"]], reac[["primaryid", "pt"]],
                      on="primaryid", how="inner")
    print(f"  Registros tras JOIN: {len(merged):,}")
    print(f"  Pares unicos farmaco-reaccion: {merged.groupby(['drugname', 'pt']).size().shape[0]:,}")

    # ── Heatmap ────────────────────────────────────────────────────────────
    top10_drugs = merged["drugname"].value_counts().head(10).index
    top10_reac  = merged["pt"].value_counts().head(10).index
    subset = merged[merged["drugname"].isin(top10_drugs) & merged["pt"].isin(top10_reac)]
    pivot = subset.groupby(["drugname", "pt"]).size().unstack(fill_value=0)

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.4, linecolor="white", ax=ax)
        ax.set_title("Heatmap: Farmaco (PS) x Reaccion Adversa\nTop 10 x Top 10",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Reaccion Adversa (MedDRA PT)")
        ax.set_ylabel("Farmaco")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        save_plot(fig, "s07_heatmap_drug_reac.png")

    # ── Top pares ─────────────────────────────────────────────────────────
    top_pares = (
        merged.groupby(["drugname", "pt"]).size()
        .reset_index(name="conteo")
        .sort_values("conteo", ascending=False)
        .head(20)
    )
    save_csv(top_pares, "s07_top20_pares_farmaco_reaccion.csv")
    print(f"\n  Top 5 pares:")
    print(top_pares.head().to_string(index=False))

    print_section("FIN SCRIPT 07")


if __name__ == "__main__":
    main()
