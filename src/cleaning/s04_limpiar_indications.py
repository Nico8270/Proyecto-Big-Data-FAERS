"""
src/cleaning/s04_limpiar_indications.py
========================================
Limpieza específica de la tabla INDI (indicaciones).

Lee de  : data/raw/faers/INDI25Q4.txt
Escribe : data/clean_data/INDI_clean.parquet
Log     : src/cleaning/logs/s04_indi_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_INDI

LOG_DIR = Path(__file__).parent / "logs"


def cargar_indi() -> pd.DataFrame:
    path = RAW_DATA / "INDI25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_indi(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # -- 1. Sin primaryid ----------------------------------------------------
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # -- 2. Normalizar indicación --------------------------------------------
    col_texto = RULES_INDI["columna_texto"]
    if col_texto in df.columns:
        df[col_texto] = df[col_texto].fillna(RULES_INDI["default_valor"]).str.upper().str.strip()
        mask_len = df[col_texto].str.len() >= RULES_INDI["pt_min_len"]
        df = df[mask_len]
        log["indicaciones_normalizadas"] = len(df)

    # -- 3. Eliminar columnas con >80% nulos ---------------------------------
    umbral = 0.8
    cols_eliminar = df.columns[df.isnull().mean() > umbral].tolist()
    df = df.drop(columns=cols_eliminar)
    log["columnas_eliminadas"] = cols_eliminar
    print(f"  Columnas eliminadas (>80% nulos): {cols_eliminar}")

    # -- 4. Duplicados -------------------------------------------------------
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s04_indi_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza INDI - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s04] Limpieza INDI")
    print("  " + "-" * 56)

    df = cargar_indi()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_indi(df)

    out = CLEAN_DATA / "INDI_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} -> {log['filas_finales']:,} "
          f"(-{log['filas_eliminadas']:,} filas, -{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
