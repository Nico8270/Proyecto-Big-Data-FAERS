"""
s08_duplicados.py
==================
Script 08 — Deteccion y reporte de duplicados en todas las tablas FAERS.

Salidas:
  • CSV con estadisticas de duplicados por tabla
  • Resumen impreso en consola
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_eda import *
from utils_eda import load_table, save_csv, print_section


def detectar_duplicados(nombre: str) -> dict:
    """Detecta duplicados completos y por primaryid en una tabla FAERS."""
    df = load_table(nombre)
    dups_completos = df.duplicated().sum()
    stats = {
        "tabla":          nombre,
        "filas":          len(df),
        "duplicados_completos": int(dups_completos),
        "pct_duplicados_completos": round(dups_completos / len(df) * 100, 4),
    }
    if "primaryid" in df.columns:
        dups_pk = df["primaryid"].duplicated().sum()
        stats["duplicados_primaryid"] = int(dups_pk)
        stats["pct_duplicados_primaryid"] = round(dups_pk / len(df) * 100, 4)
    else:
        stats["duplicados_primaryid"] = "-"
        stats["pct_duplicados_primaryid"] = "-"

    return stats


def main():
    print_section("SCRIPT 08: DETECCION DE DUPLICADOS")

    resultados = []
    for nombre in FAERS_FILES:
        try:
            stats = detectar_duplicados(nombre)
            resultados.append(stats)
            print(f"\n  [{nombre}]")
            for k, v in stats.items():
                if k != "tabla":
                    print(f"    {k:<40} {v}")
        except Exception as e:
            print(f"\n  [{nombre}] [NO DISPONIBLE] — {e}")

    if resultados:
        df_res = pd.DataFrame(resultados)
        save_csv(df_res, "s08_duplicados_por_tabla.csv")

    print_section("FIN SCRIPT 08")


if __name__ == "__main__":
    main()
