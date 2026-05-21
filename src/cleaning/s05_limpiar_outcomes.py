"""
src/cleaning/s05_limpiar_outcomes.py
======================================
Limpieza de la tabla OUTC (resultados clínicos).

Lee de  : data/raw/faers/OUTC25Q4.txt
Escribe : data/clean_data/OUTC_clean.parquet
Log     : src/cleaning/logs/s05_outc_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_OUTC

LOG_DIR = Path(__file__).parent / "logs"


def cargar_outc() -> pd.DataFrame:
    path = RAW_DATA / "OUTC25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING,
                     low_memory=False,
                     usecols=["primaryid", "outc_cod"])
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_outc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # ── 1. Sin primaryid ────────────────────────────────────────────────────
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # ── 2. Normalizar outc_cod ──────────────────────────────────────────────
    df["outc_cod"] = df["outc_cod"].fillna(RULES_OUTC["default_label"])
    df["outc_cod"] = df["outc_cod"].str.upper().str.strip()

    mask_valida = df["outc_cod"].isin(RULES_OUTC["validos"])
    reemplazados = int((~mask_valida).sum())
    df.loc[~mask_valida, "outc_cod"] = RULES_OUTC["default_label"]
    log["outc_cod_normalizados"] = reemplazados
    print(f"  outc_cod no reconocidos reemplazados por '{RULES_OUTC['default_label']}': {reemplazados:,}")

    # ── 3. Duplicados por primaryid ─────────────────────────────────────────
    dups = df.duplicated(subset=["primaryid"]).sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados por primaryid eliminados: {dups:,}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s05_outc_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza OUTC — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s05] Limpieza OUTC")
    print("  " + "─" * 56)

    df = cargar_outc()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_outc(df)

    out = CLEAN_DATA / "OUTC_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} → {log['filas_finales']:,} "
          f"(−{log['filas_eliminadas']:,} filas, −{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
