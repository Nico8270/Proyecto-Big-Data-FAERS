"""
src/cleaning/s08_consolidar_clean.py
======================================
Une todas las tablas limpias en un solo dataset consolidado.

Entrada : data/clean_data/  → archivos *_clean.parquet (s07 tablas)
Salida  : data/clean_data/dataset_consolidado.parquet
Log     : src/cleaning/logs/s08_consol_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import CLEAN_DATA

LOG_DIR  = Path(__file__).parent / "logs"
CLEAN    = CLEAN_DATA

TABLAS = {
    "DEMO":  "DEMO_clean.parquet",
    "DRUG":  "DRUG_clean.parquet",
    "REAC":  "REAC_clean.parquet",
    "OUTC":  "OUTC_clean.parquet",
    "INDI":  "INDI_clean.parquet",
    "THER":  "THER_clean.parquet",
    "RPSR":  "RPSR_clean.parquet",
}


def cargar_tablas() -> dict:
    tablas = {}
    for nombre, archivo in TABLAS.items():
        path = CLEAN / archivo
        if not path.exists():
            print(f"  [AVISO] No encontrado: {path}")
            continue
        tablas[nombre] = pq.read_table(path).to_pandas()
        print(f"  {nombre:<6} {len(tablas[nombre]):>10,} filas × {tablas[nombre].shape[1]} columnas")
    return tablas


def consolidar(tablas: dict) -> pd.DataFrame:
    """Inner join por primaryid de todas las tablas."""
    print("\n  Ejecutando JOIN por primaryid...")

    resultado = tablas["DEMO"]
    for nombre, df in tablas.items():
        if nombre == "DEMO":
            continue
        antes = len(resultado)
        resultado = pd.merge(resultado, df, on="primaryid", how="inner", suffixes=("", f"_{ nombre.lower() }"))
        despues = len(resultado)
        eliminados = antes - despues
        if eliminados > 0:
            print(f"    {nombre}: perdidos {eliminados:,} registros en el JOIN (quedan {despues:,})")

    print(f"\n  Registros finales tras JOIN: {len(resultado):,}")
    return resultado


def validar(resultado: pd.DataFrame):
    dup_pk = resultado["primaryid"].duplicated().sum()
    print(f"\n  primaryid únicos: {resultado['primaryid'].nunique():,}")
    print(f"  primaryid duplicados: {dup_pk:,}")
    if dup_pk > 0:
        print("  [AVISO] Hay primaryid duplicados — revisar la consolidación.")


def guardar_log(tablas: dict, resultado: pd.DataFrame):
    path = LOG_DIR / f"s08_consol_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Consolidación — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        f.write("  Por tabla:\n")
        for nombre, df in tablas.items():
            f.write(f"    {nombre:<10} {len(df):>10,} filas\n")
        f.write(f"\n  Consolidado: {len(resultado):,} filas × {resultado.shape[1]} columnas\n")
        f.write(f"  primaryid únicos: {resultado['primaryid'].nunique():,}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s08] Consolidar tablas limpias")
    print("  " + "─" * 56)

    tablas = cargar_tablas()
    if not tablas:
        print("  [ERROR] No se encontraron tablas limpias.")
        return

    df = consolidar(tablas)
    validar(df)

    out = CLEAN / "dataset_consolidado.parquet"
    pq.write_table(pa.Table.from_pandas(df), out, compression="snappy")
    print(f"\n  [OK] Guardado: {out} ({len(df):,} filas)")

    guardar_log(tablas, df)
    print(f"  Resumen: {sum(len(v) for v in tablas.values()):,} registros individuales → "
          f"{len(df):,} registros consolidados\n")


if __name__ == "__main__":
    main()
