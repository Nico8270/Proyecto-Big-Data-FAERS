"""
utils_eda.py
============
Funciones reutilizables para todos los scripts EDA.
Evita duplicacion de logica de carga, guardado y formateo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config_eda import (
    DATA_DIR, OUTPUT_DIR, FAERS_FILES,
    FAERS_SEP, FAERS_ENCODING, FIGURE_DPI,
    PALETTE_CATEG, PALETTE_SEQUEN,
)


# ── Carga de datos ───────────────────────────────────────────────────────────

def load_table(name: str, usecols: list = None, nrows: int = None) -> pd.DataFrame:
    """Carga una tabla FAERS por clave (DEMO, DRUG, REAC, ...)."""
    filename = FAERS_FILES.get(name)
    if not filename:
        raise KeyError(f"Tabla desconocida: {name}")
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No encontrado: {filepath}")
    df = pd.read_csv(
        filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING,
        low_memory=False, usecols=usecols, nrows=nrows,
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


# ── Guardado de salidas ──────────────────────────────────────────────────────

def save_plot(fig, filename: str):
    """Guarda una figura en outputs/eda_results/ y la cierra."""
    fig.savefig(OUTPUT_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


def save_csv(df: pd.DataFrame, filename: str):
    """Guarda un DataFrame como CSV en outputs/eda_results/."""
    df.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"  [OK] {filename}")


# ── Formateo de consola ─────────────────────────────────────────────────────

def print_section(titulo: str, ancho: int = 70):
    barra = "─" * ancho
    print(f"\n{barra}")
    print(f"  {titulo}")
    print(f"{barra}")


def print_header_columnas(columnas: list, anchos: list):
    """Imprime el encabezado alineado de una tabla de consola."""
    linea = "  " + "  ".join(f"{c:<{a}}" for c, a in zip(columnas, anchos))
    print(linea)
    print("  " + "  ".join("─" * (a - 2) for a in anchos))


def print_fila_columnas(valores: list, anchos: list):
    """Imprime una fila alineada."""
    linea = "  " + "  ".join(f"{str(v)[:a-2]:<{a}}" for v, a in zip(valores, anchos))
    print(linea)


# ── Estadisticas basicas por tabla ──────────────────────────────────────────

def stats_basicas(nombre: str) -> dict:
    """Devuelve estadisticas basicas de una tabla FAERS."""
    df = load_table(nombre)
    return {
        "tabla":          nombre,
        "filas":          len(df),
        "columnas":       df.shape[1],
        "nulos":          int(df.isnull().sum().sum()),
        "pct_nulos":      round(df.isnull().sum().sum() / (len(df) * df.shape[1]) * 100, 2),
        "duplicados":     int(df.duplicated().sum()),
        "pct_duplicados": round(df.duplicated().sum() / len(df) * 100, 2),
        "pk_unicos":      int(df["primaryid"].nunique()) if "primaryid" in df.columns else 0,
    }
