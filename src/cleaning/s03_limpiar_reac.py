"""
src/cleaning/s03_limpiar_reac.py
=================================
Limpieza específica de la tabla REAC (reacciones adversas).

Lee de  : data/raw/faers/REAC25Q4.txt
Escribe : data/clean_data/REAC_clean.parquet
Log     : src/cleaning/logs/s03_reac_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_REAC

LOG_DIR = Path(__file__).parent / "logs"


def cargar_reac() -> pd.DataFrame:
    path = RAW_DATA / "REAC25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING,
                     low_memory=False,
                     usecols=["primaryid", "caseid", "pt", "drug_rec_act"])
    df.columns = df.columns.str.strip().str.lower()
    return df


def normalizar_pt(val: str) -> str:
    """Aplica mapeo de sinónimos y normaliza a Title Case."""
    if not val or not isinstance(val, str):
        return val
    v = val.upper().strip()
    return RULES_REAC["sinonimos"].get(v, v.title().strip())


def limpiar_reac(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # ── 1. Sin primaryid ────────────────────────────────────────────────────
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # ── 2. Normalizar reacción MedDRA ───────────────────────────────────────
    df["pt"] = df["pt"].apply(normalizar_pt)

    # Filtrar PT ruidosos (longitud < umbral)
    mask_pt = df["pt"].notna() & (df["pt"].str.len() >= RULES_REAC["pt_min_len"])
    eliminados_pt = (~mask_pt).sum()
    df = df[mask_pt]
    log["pt_ruidosos_eliminados"] = int(eliminados_pt)
    print(f"  PT ruidosos eliminados (<{RULES_REAC['pt_min_len']} caracteres): {eliminados_pt:,}")

    # ── 3. Normalizar drug_rec_act ──────────────────────────────────────────
    if "drug_rec_act" in df.columns:
        df["drug_rec_act"] = df["drug_rec_act"].fillna("U").str.upper().str.strip()
        log["drug_rec_act_normalizados"] = int((df["drug_rec_act"] == "U").sum())

    # ── 4. Duplicados ───────────────────────────────────────────────────────
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s03_reac_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza REAC — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s03] Limpieza REAC")
    print("  " + "─" * 56)

    df = cargar_reac()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_reac(df)

    out = CLEAN_DATA / "REAC_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} → {log['filas_finales']:,} "
          f"(−{log['filas_eliminadas']:,} filas, −{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
