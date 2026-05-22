"""
src/cleaning/s07_limpiar_reporter.py
=====================================
Limpieza específica de la tabla RPSR (reporter).

Lee de  : data/raw/faers/RPSR25Q4.txt
Escribe : data/clean_data/RPSR_clean.parquet
Log     : src/cleaning/logs/s07_rpsr_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING

LOG_DIR = Path(__file__).parent / "logs"


def cargar_rpsr() -> pd.DataFrame:
    path = RAW_DATA / "RPSR25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_rpsr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # -- 1. Sin primaryid ----------------------------------------------------
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # -- 2. Normalizar rept_cod (rol del reportador) -------------------------
    if "rept_cod" in df.columns:
        df["rept_cod"] = df["rept_cod"].fillna("UNK").str.upper().str.strip()
        log["rept_cod_normalizados"] = int((df["rept_cod"] == "UNK").sum())

    # -- 3. Eliminar columnas con >95% nulos ---------------------------------
    umbral = 0.95
    cols_eliminar = df.columns[df.isnull().mean() > umbral].tolist()
    df = df.drop(columns=cols_eliminar)
    log["columnas_eliminadas"] = cols_eliminar
    print(f"  Columnas eliminadas (>95% nulos): {cols_eliminar}")

    # -- 4. Duplicados -------------------------------------------------------
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s07_rpsr_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza RPSR - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s07] Limpieza RPSR")
    print("  " + "-" * 56)

    df = cargar_rpsr()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_rpsr(df)

    out = CLEAN_DATA / "RPSR_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} -> {log['filas_finales']:,} "
          f"(-{log['filas_eliminadas']:,} filas, -{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
