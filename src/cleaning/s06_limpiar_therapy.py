"""
src/cleaning/s06_limpiar_therapy.py
=====================================
Limpieza de la tabla THER (terapia).

Lee de  : data/raw/faers/THER25Q4.txt
Escribe : data/clean_data/THER_clean.parquet
Log     : src/cleaning/logs/s06_ther_YYYY-MM-DD.txt
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


def cargar_ther() -> pd.DataFrame:
    path = RAW_DATA / "THER25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_ther(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # ── 1. Sin primaryid ────────────────────────────────────────────────────
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # ── 2. drug_seq como entero positivo ────────────────────────────────────
    df["drug_seq"] = pd.to_numeric(df["drug_seq"], errors="coerce")
    df.loc[df["drug_seq"] <= 0, "drug_seq"] = pd.NA
    df["drug_seq"] = df["drug_seq"].fillna(0).astype(int)
    log["drug_seq_rellenados"] = int((df["drug_seq"] == 0).sum())

    # ── 3. Duplicados completos ─────────────────────────────────────────────
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    # ── 4. Columnas con todos los valores vacíos ────────────────────────────
    cols_vacias = df.columns[df.notna().sum() == 0].tolist()
    if cols_vacias:
        df = df.drop(columns=cols_vacias)
        log["columnas_eliminadas_vacias"] = cols_vacias
        print(f"  Columnas vacías eliminadas: {cols_vacias}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s06_ther_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza THER — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s06] Limpieza THER")
    print("  " + "─" * 56)

    df = cargar_ther()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_ther(df)

    out = CLEAN_DATA / "THER_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} → {log['filas_finales']:,} "
          f"(−{log['filas_eliminadas']:,} filas, −{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
