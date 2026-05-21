"""
src/cleaning/s01_limpiar_demo.py
=================================
Limpieza específica de la tabla DEMO (demografía de pacientes).

Lee de  : data/raw/faers/DEMO25Q4.txt
Escribe : data/clean_data/DEMO_clean.parquet
Log     : src/cleaning/logs/s01_demo_YYYY-MM-DD.txt
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_DEMO


LOG_DIR  = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def cargar_demo() -> pd.DataFrame:
    """Carga el DEMO crudo."""
    path = RAW_DATA / "DEMO25Q4.txt"
    df = pd.read_csv(path, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def limpiar_demo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica todas las reglas de limpieza sobre DEMO."""
    log = {"filas_iniciales": len(df)}
    antes = len(df)

    # ── 1. Eliminar sin primaryid ──────────────────────────────────────────
    mask = df["primaryid"].notna()
    eliminados = (~mask).sum()
    df = df[mask].copy()
    log["eliminados_sin_primaryid"] = int(eliminados)
    print(f"  Sin primaryid eliminados: {eliminados:,}")

    # ── 2. Normalizar sexo ──────────────────────────────────────────────────
    df["sex"] = df["sex"].fillna(RULES_DEMO["sexo_default"])
    df.loc[~df["sex"].isin(RULES_DEMO["sexo_validos"]), "sex"] = RULES_DEMO["sexo_default"]
    log["sex_normalizados"] = int((df["sex"] == RULES_DEMO["sexo_default"]).sum())
    print(f"  Sexo normalizado: {log['sex_normalizados']:,}")

    # ── 3. Limpiar edad ────────────────────────────────────────────────────
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < RULES_DEMO["edad_min"]) | (df["age"] > RULES_DEMO["edad_max"]), "age"] = pd.NA
    log["edad_fuera_rango"]   = int(df["age"].isna().sum())
    log["edad_imputada_mediana"] = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(df["age"].median())
    imputadas = int(df["age"].isna().sum())  # quedan solo si TODAS son NaN (imposible)
    log["edad_final_nulos"] = imputadas
    print(f"  Edad fuera de rango / imputada: {log['edad_fuera_rango']:,}")

    # ── 4. Normalizar paises (mayúsculas + strip) ───────────────────────────
    if "reporter_country" in df.columns:
        df["reporter_country"] = df["reporter_country"].str.upper().str.strip()
        log["paises_normalizados"] = int((df["reporter_country"].str.len() != 2).sum())
        print(f"  Países normalizados: {log['paises_normalizados']:,}")

    # ── 5. Eliminar columnas con > threshold de nulos ──────────────────────
    umbral = RULES_DEMO["drop_columns_threshold_pct"] / 100
    pct_nulos_por_col = df.isnull().mean()
    cols_eliminar = pct_nulos_por_col[pct_nulos_por_col > umbral].index.tolist()
    df = df.drop(columns=cols_eliminar)
    log["columnas_eliminadas"] = cols_eliminar
    print(f"  Columnas eliminadas (>60% nulos): {cols_eliminar}")

    # ── 6. Eliminar duplicados completos ───────────────────────────────────
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log["duplicados_eliminados"] = int(dups)
    print(f"  Duplicados eliminados: {dups:,}")

    log["filas_finales"] = len(df)
    log["filas_eliminadas"] = antes - len(df)
    return df, log


def guardar_log(log: dict, nombre_archivo: str):
    """Guarda el log de limpieza en src/cleaning/logs/."""
    path = LOG_DIR / nombre_archivo
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza DEMO — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s01] Limpieza DEMO")
    print("  " + "─" * 56)

    df = cargar_demo()
    print(f"  Cargado: {len(df):,} filas × {df.shape[1]} columnas")

    df_clean, log = limpiar_demo(df)

    # Guardar Parquet
    out = CLEAN_DATA / "DEMO_clean.parquet"
    table = pa.Table.from_pandas(df_clean)
    pq.write_table(table, out, compression="snappy")
    print(f"  [OK] Guardado: {out} ({len(df_clean):,} filas)")

    guardar_log(log, f"s01_demo_{datetime.now():%Y-%m-%d}.txt")
    print(f"\n  Resumen: {log['filas_iniciales']:,} → {log['filas_finales']:,} "
          f"(−{log['filas_eliminadas']:,} filas, −{log['filas_eliminadas']/log['filas_iniciales']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
