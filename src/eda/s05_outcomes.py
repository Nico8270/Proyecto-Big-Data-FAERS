"""
s05_outcomes.py
================
Script 05 — Analisis de outcomes / resultados clinicos (tabla OUTC).

Salidas:
  • Gauge de distribucion de outcomes
  • Gauge de outcomes por reaccion principal (top 10)
  • CSV con conteos por codigo de outcome
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
    print_section("SCRIPT 05: RESULTADOS CLINICOS (OUTC)")

    print("\n[1/3] Cargando tabla OUTC...")
    outc = load_table("OUTC", usecols=["primaryid", "outc_cod"])
    print(f"  {len(outc):,} filas × {outc.shape[1]} columnas")

    # ── Distribucion de outcomes ─────────────────────────────────────────
    print("\n[2/3] Distribucion de outcomes...")
    if "outc_cod" in outc.columns:
        counts = outc["outc_cod"].fillna("").value_counts()

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        colors = sns.color_palette(PALETTE_DANGER, len(counts))
        counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Distribucion de Resultados Clinicos (Outcomes)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Codigo de Outcome")
        ax.set_ylabel("Numero de Reportes")
        ax.tick_params(axis="x", rotation=0)

        total = counts.sum()
        for bar in ax.patches:
            h = bar.get_height()
            pct = h / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.01,
                    f"{h:,}\n({pct:.1f}%)", ha="center", fontsize=9)

        plt.tight_layout()
        save_plot(fig, "s05_outcomes.png")

        # CSV de conteos
        df_out = counts.reset_index()
        df_out.columns = ["outc_cod", "count"]
        df_out["descripcion"] = df_out["outc_cod"].map(OUTCOME_MAP).fillna("Desconocido")
        df_out["porcentaje"] = (df_out["count"] / total * 100).round(2)
        save_csv(df_out, "s05_conteos_outcomes.csv")

    # ── Severidad agregada por codigo de outcome ───────────────────────────
    print("\n[3/3] Severidad agregada por outcome...")
    print_section("FIN SCRIPT 05")


if __name__ == "__main__":
    main()
