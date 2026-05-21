"""
s02_demograficos.py
====================
Script 02 — Analisis demografico de la tabla DEMO.

Salidas:
  • Gauge 3×1: sexo, distribucion edad, top 10 paises
  • Boxplot edad por genero
  • Estadisticas de edad guardadas en CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_eda import *
from utils_eda import load_table, save_plot, save_csv, print_section


def cargar_demo():
    return load_table("DEMO")


def gauge_sexo(demo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    counts = demo["sex"].fillna("U").value_counts()
    colors = sns.color_palette(PALETTE_CATEG, len(counts))
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Distribucion por Genero", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genero (F=Femenino, M=Masculino, U=Desconocido)")
    ax.set_ylabel("Numero de Reportes")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=10)
    plt.tight_layout()
    save_plot(fig, "s02_genero.png")


def gauge_edad(demo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    edades = pd.to_numeric(demo["age"], errors="coerce")
    edades = edades[(edades >= 0) & (edades <= 120)]
    ax.hist(edades.dropna(), bins=40, color="teal", edgecolor="white", alpha=0.85)
    ax.set_title("Distribucion de Edad", fontsize=14, fontweight="bold")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    save_plot(fig, "s02_edad.png")


def gauge_paises(demo: pd.DataFrame, top_n: int = 10):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    top = demo["reporter_country"].value_counts().head(top_n)
    colors = sns.color_palette(PALETTE_SEQUEN, len(top))
    top.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Paises Reportantes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Numero de Reportes")
    ax.invert_yaxis()
    for i, v in enumerate(top):
        ax.text(v + max(top) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    plt.tight_layout()
    save_plot(fig, "s02_paises.png")


def boxplot_edad_genero(demo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    data = demo.copy()
    data["age_num"] = pd.to_numeric(data["age"], errors="coerce")
    data = data[(data["age_num"] >= 0) & (data["age_num"] <= 120)]
    sns.boxplot(data=data, x="sex", y="age_num", ax=ax, palette=PALETTE_CATEG)
    ax.set_title("Edad por Genero", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genero")
    ax.set_ylabel("Edad (años)")
    plt.tight_layout()
    save_plot(fig, "s02_boxplot_edad_genero.png")

    # Estadisticas CSV
    stats = data.groupby("sex")["age_num"].describe().round(2)
    save_csv(stats.reset_index(), "s02_estadisticas_edad_por_genero.csv")


def main():
    print_section("SCRIPT 02: ANALISIS DEMOGRAFICOS")

    print("\n[1/5] Cargando tabla DEMO...")
    demo = cargar_demo()
    print(f"  {len(demo):,} filas × {demo.shape[1]} columnas")

    print("\n[2/5] Distribucion por genero...")
    gauge_sexo(demo)

    print("\n[3/5] Distribucion de edad...")
    gauge_edad(demo)

    print("\n[4/5] Top paises reportantes...")
    gauge_paises(demo)

    print("\n[5/5] Boxplot edad x genero...")
    boxplot_edad_genero(demo)

    print_section("FIN SCRIPT 02")


if __name__ == "__main__":
    main()
