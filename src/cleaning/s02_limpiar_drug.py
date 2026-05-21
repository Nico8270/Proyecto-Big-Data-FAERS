"""
src/cleaning/s02_limpiar_drug.py
=================================
Limpieza específica de la tabla DRUG (medicamentos).

Lee de  : data/raw/faers/DRUG25Q4.txt
Escribe : data/clean_data/DRUG_clean.parquet
Log     : src/cleaning/logs/s02_drug_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_DRUG

LOG_DIR = Path(__file__).parent / "logs"


def cargar_drug() -> pd.DataFrame:
    path = RAW_DATA / "DRUG25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_drug(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # ── 1. Sin primaryid ────────────────────────────────────────────────────
    mask = df["primaryid"].notna()
    df = df[mask].copy()
    log["sin_primaryid"] = int((~mask).sum())

    # ── 2. Sin drugname ─────────────────────────────────────────────────────
    mask = df["drugname"].notna()
    df = df[mask].copy()
    log["sin_drugname"] = int((~mask).sum())

    # ── 3. Normalizar drugname (UPPER + strip) ──────────────────────────────
    df["drugname"] = df["drugname"].str.upper().str.strip()
    log["drugname_normalizados"] = len(df)

    # ── 4. Normalizar role_cod ──────────────────────────────────────────────
    df["role_cod"] = df["role_cod"].fillna(RULES_DRUG["rol_default"])
    df.loc[~df["role_cod"].isin(RULES_DRUG["rol_validos"]), "role_cod"] = RULES_DRUG["rol_default"]
    log["role_cod_normalizados"] = int((df["role_cod"] == RULES_DRUG["rol_default"]).sum())

    # ── 5. Normalizar route ─────────────────────────────────────────────────
    df["route"] = df["route"].fillna(RULES_DRUG["ruta_default"])
    df["route"] = df["route"].str.upper().str.strip()
    log["route_normalizados"] = int((df["route"] == RULES_DRUG["ruta_default"]).sum())

    # ── 6. Limpiar dosis ────────────────────────────────────────────────────
    df["dose_amt"] = pd.to_numeric(df["dose_amt"], errors="coerce")
    fuera_rango = (df["dose_amt"] < RULES_DRUG["dosis_min"]) | (df["dose_amt"] > RULES_DRUG["dosis_max"])
    df.loc[fuera_rango, "dose_amt"] = pd.NA
    log["dosis_fuera_rango"] = int(fuera_rango.sum())
    mediana_dosis = df["dose_amt"].median()
    df["dose_amt"] = df["dose_amt"].fillna(mediana_dosis)
    log["dosis_imputadas_mediana"] = log["dosis_fuera_rango"]
    print(f"  Dosis normalizadas (mediana={mediana_dosis:.1f}): {log['dosis_fuera_rango']:,}")

    # ── 7. Normalizar dechal / rechal ───────────────────────────────────────
    for col in ("dechal", "rechal"):
        if col in df.columns:
            df[col] = df[col].fillna("U")
            df.loc[~df[col].isin(RULES_DRUG["dechal_validos"]), col] = "U"
            log[f"{col}_normalizados"] = int((df[col] == "U").sum())

    # ── 8. Eliminar columnas con >90% nulos ─────────────────────────────────
    umbral = RULES_DRUG["drop_columns_threshold_pct"] / 100
    cols_eliminar = df.columns[df.isnull().mean() > umbral].tolist()
    df = df.drop(columns=cols_eliminar)
    log["columnas_eliminadas"] = cols_eliminar
    print(f"  Columnas eliminadas (>90% nulos): {cols_eliminar}")

    # ── 9. Duplicados ───────────────────────────────────────────────────────
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    log["filas_finales"]    = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict):
    path = LOG_DIR / f"s02_drug_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza DRUG — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s02] Limpieza DRUG")
    print("  " + "─" * 56)

    df = cargar_drug()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_drug(df)

    out = CLEAN_DATA / "DRUG_clean.parquet"
    pq.write_table(pa.Table.from_pandas(df_clean), out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log)
    print(f"\n  Resumen: {log['filas_iniciales']:,} → {log['filas_finales']:,} "
          f"(−{log['filas_eliminadas']:,} filas, −{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
