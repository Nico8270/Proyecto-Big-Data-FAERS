"""
s03_drogas.py
=============
Script 03 — Analisis de medicamentos (tabla DRUG).

Salidas:
  • Top 25 farmacos mas reportados
  • Distribucion por rol del farmaco (PS/SS/C/I)
  • Top 15 vias de administracion
  • Distribucion de dosis
  • CSV con estadisticas de roles
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
    print_section("SCRIPT 03: ANALISIS DE MEDICAMENTOS")

    print("\n[1/5] Cargando tabla DRUG...")
    drug = load_table("DRUG")
    print(f"  {len(drug):,} filas × {drug.shape[1]} columnas")

    # ── Top farmacos ────────────────────────────────────────────────────────
    print("\n[2/5] Top 25 farmacos mas reportados...")
    if "drugname" in drug.columns:
        top = (drug["drugname"].str.upper().str.strip().value_counts().head(25))
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = sns.color_palette(PALETTE_MAIN, len(top))
        top.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Top 25 Farmacos mas Reportados", fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Registros")
        ax.set_ylabel("Nombre del Farmaco")
        ax.invert_yaxis()
        plt.tight_layout()
        save_plot(fig, "s03_top25_farmacos.png")
        save_csv(top.rename("reportes").reset_index().rename(columns={"index": "farmaco"}),
                 "s03_top25_farmacos.csv")

    # ── Roles ───────────────────────────────────────────────────────────────
    print("\n[3/5] Distribucion por rol del farmaco...")
    if "role_cod" in drug.columns:
        counts = drug["role_cod"].fillna("??").value_counts()
        labels = [DRUG_ROLE_MAP.get(c, c) for c in counts.index]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        bars = counts.plot(kind="bar", ax=ax1, color=sns.color_palette(PALETTE_CATEG, len(counts)), edgecolor="white")
        ax1.set_title("Distribucion por Rol del Farmaco", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Registros")
        ax1.tick_params(axis="x", rotation=0)
        for bar, label in zip(bars.patches, labels):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.02,
                     f"{h:,}\n({h / len(drug) * 100:.1f}%)", ha="center", fontsize=9)
        ax1.set_xticklabels(labels, rotation=0)

        ax2.pie(counts, labels=labels, autopct="%1.1f%%",
                colors=sns.color_palette(PALETTE_CATEG, len(counts)),
                startangle=90, wedgeprops={"edgecolor": "white"})
        ax2.set_title("Proporcion por Rol", fontsize=13, fontweight="bold")
        ax2.axis("equal")
        plt.tight_layout()
        save_plot(fig, "s03_roles_farmaco.png")

        role_stats = counts.reset_index()
        role_stats.columns = ["role_cod", "count"]
        role_stats["descripcion"] = role_stats["role_cod"].map(DRUG_ROLE_MAP)
        role_stats["porcentaje"] = (role_stats["count"] / len(drug) * 100).round(2)
        save_csv(role_stats, "s03_estadisticas_roles.csv")

    # ── Vias de administracion ──────────────────────────────────────────────
    print("\n[4/5] Vias de administracion...")
    if "route" in drug.columns:
        top_routes = drug["route"].fillna("UNK").value_counts().head(15)
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        colors = sns.color_palette(PALETTE_SEQUEN, len(top_routes))
        top_routes.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Top 15 Vias de Administracion", fontsize=14, fontweight="bold")
        ax.set_xlabel("Numero de Registros")
        ax.invert_yaxis()
        for i, v in enumerate(top_routes):
            ax.text(v + max(top_routes) * 0.01, i, f"{v:,}", va="center", fontsize=9)
        plt.tight_layout()
        save_plot(fig, "s03_vias_administracion.png")

    # ── Dosis ───────────────────────────────────────────────────────────────
    print("\n[5/5] Distribucion de dosis...")
    if "dose_amt" in drug.columns:
        dosis = pd.to_numeric(drug["dose_amt"], errors="coerce")
        dosis = dosis[(dosis > 0) & (dosis < 10000)]
        if len(dosis) > 0:
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.hist(dosis.dropna(), bins=50, color="teal", edgecolor="white", alpha=0.85)
            ax.set_title("Distribucion de Dosis Reportadas", fontsize=13, fontweight="bold")
            ax.set_xlabel("Dosis (unidades)")
            ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            save_plot(fig, "s03_dosis.png")

    print_section("FIN SCRIPT 03")


if __name__ == "__main__":
    main()
