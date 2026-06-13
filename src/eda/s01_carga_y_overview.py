"""
s01_carga_y_overview.py
========================
Script 01 — Carga de todas las tablas FAERS y descripcion general.

Salidas:
  • Grafico de nulos por columna por tabla
  • CSV con informacion de cada columna (tipo, nulos, %)
  • Resumen de carga en consola
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
    print_section("SCRIPT 01: CARGA Y DESCRIPCION GENERAL")

    tablas = {}
    for nombre in FAERS_FILES:
        try:
            tablas[nombre] = load_table(nombre)
            print(f"  {nombre:<6} {len(tablas[nombre]):>12,} filas × {tablas[nombre].shape[1]} columnas")
        except Exception as e:
            print(f"  {nombre:<6} [NO DISPONIBLE] — {e}")

    total_filas = sum(len(df) for df in tablas.values())
    print(f"\n  Total aproximado: {total_filas:,} registros en {len(tablas)} tablas")

    # ── Informacion de columnas por tabla ─────────────────────────────────
    print_section("Informacion de columnas por tabla")

    todas_las_columnas = []
    for nombre, df in tablas.items():
        print(f"\n  [{nombre}]")
        info = pd.DataFrame({
            "columna":   df.columns,
            "tipo":      df.dtypes.astype(str).values,
            "nulos":     df.isnull().sum().values,
            "pct_nulos": (df.isnull().sum() / len(df) * 100).round(2).values,
        })
        print(f"  {'columna':<35} {'tipo':<12} {'nulos':>8}  {'pct_nulos':>10}")
        print(f"  {'-'*35} {'-'*12} {'-'*8}  {'-'*10}")
        for _, row in info.iterrows():
            print(f"  {row.columna:<35} {row.tipo:<12} {row.nulos:>8,}  {row.pct_nulos:>9.2f}%")

        save_csv(info, f"s01_{nombre.lower()}_columnas.csv")
        todas_las_columnas.append(info.assign(tabla=nombre))

    # ── Grafico de nulos por tabla ─────────────────────────────────────────
    n_plots = len(tablas)
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for ax, (nombre, df) in zip(axes, tablas.items()):
            pct_nulos = df.isnull().mean().sort_values(ascending=False)
            pct_nulos_con_datos = pct_nulos[pct_nulos > 0]

            if not pct_nulos_con_datos.empty:
                pct_nulos_con_datos.plot(
                    kind="bar", ax=ax, color="steelblue", edgecolor="white",
                )
            else:
                ax.text(
                    0.5, 0.5,
                    "Sin valores nulos",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(f"Nulos por columna — {nombre}", fontsize=13, fontweight="bold")
            ax.set_ylabel("Proporcion de nulos")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("")

        plt.tight_layout()
        save_plot(fig, "s01_nulos_por_tabla.png")


if __name__ == "__main__":
    main()
