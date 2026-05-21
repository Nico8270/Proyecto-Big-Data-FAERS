"""
Script 01: Carga y Descripción General del Dataset FAERS
Objetivo: Cargar todos los archivos FAERS y generar estadísticas básicas
"""
import pandas as pd
from pathlib import Path
import sys

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

def load_faers_file(short_name: str, nrows: int = None) -> pd.DataFrame:
    """
    Carga un archivo FAERS y devuelve un DataFrame.
    """
    filename = FAERS_FILES[short_name]
    filepath = DATA_DIR / filename

    if not filepath.exists():
        print(f"[ADVERTENCIA] Archivo no encontrado: {filepath}")
        return pd.DataFrame()

    print(f"  Cargando {short_name}: {filename}...", end=" ", flush=True)

    try:
        df = pd.read_csv(
            filepath,
            sep=FAERS_SEP,
            encoding=FAERS_ENCODING,
            low_memory=False,
            nrows=nrows
        )
        df.columns = df.columns.str.strip().str.lower()
        print(f"{df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("  SCRIPT 01: CARGA Y DESCRIPCIÓN GENERAL – FAERS Q4 2025")
    print("=" * 70)

    # Cargar todos los archivos
    dataframes = {}
    for name in FAERS_FILES.keys():
        dataframes[name] = load_faers_file(name)

    print("\n" + "=" * 70)
    print("  RESUMEN DE CARGA")
    print("=" * 70)

    total_rows = 0
    for name, df in dataframes.items():
        if df.empty:
            print(f"  {name:6s}: [NO DISPONIBLE]")
        else:
            total_rows += len(df)
            print(f"  {name:6s}: {len(df):>12,} filas  |  {df.shape[1]:>3} columnas")

    print(f"\n  Total aproximado de registros: {total_rows:,}")

    # Guardar resumen en CSV
    summary_data = []
    for name, df in dataframes.items():
        if not df.empty:
            summary_data.append({
                "tabla": name,
                "filas": len(df),
                "columnas": df.shape[1],
                " memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / "01_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  [OK] Resumen guardado en: {summary_path}")

    # Análisis de claves primarias
    print("\n" + "=" * 70)
    print("  ANÁLISIS DE CLAVES PRIMARIAS")
    print("=" * 70)

    for name, df in dataframes.items():
        if df.empty or "primaryid" not in df.columns:
            continue

        unique_primary = df["primaryid"].nunique()
        duplicates = len(df) - unique_primary

        print(f"\n  {name}:")
        print(f"    primaryid únicos : {unique_primary:,}")
        print(f"    primaryid duplicados: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

        if "caseid" in df.columns:
            unique_case = df["caseid"].nunique()
            print(f"    caseid únicos    : {unique_case:,}")

    print("\n" + "=" * 70)
    print("  Información de columnas por tabla")
    print("=" * 70)

    for name, df in dataframes.items():
        if df.empty:
            continue

        print(f"\n── {name} ──")
        info_df = pd.DataFrame({
            "columna": df.columns,
            "tipo": df.dtypes.astype(str).values,
            "nulos": df.isnull().sum().values,
            "%_nulos": (df.isnull().sum() / len(df) * 100).round(2).values
        })
        print(info_df.to_string(index=False))

        # Guardar info detallada
        info_path = OUTPUT_DIR / f"01_{name.lower()}_columns_info.csv"
        info_df.to_csv(info_path, index=False)

    print(f"\n[OK] Información de columnas guardada en outputs/eda_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
