"""
src/hadoop/training/01_data_preparation.py
============================================
ETAPA 01: PREPARACIÓN DE DATOS PARA ENTRENAMIENTO

Lee el dataset balanceado de Machine Learning desde data/balanced_data/{tecnica}/dataset.parquet
(generado en la opción 4 de balanceo). Si no existe, advierte al usuario.
Convierte los datos balanceados a JSON Lines (JSONL) en data/staging/training_input/ para
el pipeline de Machine Learning predictivo.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BALANCED_DIR = PROJECT_ROOT / "data" / "balanced_data"
STAGING_DIR  = PROJECT_ROOT / "data" / "staging" / "training_input"

def main():
    print("\n" + "=" * 60)
    print("  ETAPA 01: PREPARACIÓN DE DATOS BALANCEADOS")
    print("=" * 60)

    # Buscar dinámicamente un dataset balanceado en las subcarpetas de balanced_data
    input_parquet = None
    if BALANCED_DIR.exists():
        for subfolder in BALANCED_DIR.iterdir():
            if subfolder.is_dir():
                parquet_file = subfolder / "dataset.parquet"
                if parquet_file.exists():
                    input_parquet = parquet_file
                    print(f"  [INFO] Detectado dataset balanceado por la técnica '{subfolder.name}'.")
                    break

    # Si no se encontró en balanced_data, buscar el dataset consolidado original como fallback
    if not input_parquet:
        fallback_parquet = PROJECT_ROOT / "data" / "clean_data" / "dataset_consolidado.parquet"
        if fallback_parquet.exists():
            input_parquet = fallback_parquet
            print("  [AVISO] No se encontró un dataset balanceado en data/balanced_data/.")
            print(f"          Usando el dataset consolidado original: {fallback_parquet.name}")
        else:
            print(f"\n[ERROR] No se encontró ningún dataset válido para entrenar.")
            print("  Por favor, ejecute primero la opción 4 (Balancear datos) o la opción 3 (Join con Hadoop).")
            sys.exit(1)

    # Limpiar y recrear el staging
    if STAGING_DIR.exists():
        try:
            shutil.rmtree(STAGING_DIR)
        except OSError as e:
            print(f"  [AVISO] No se pudo limpiar {STAGING_DIR}: {e}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Leyendo dataset de entrenamiento: {input_parquet.relative_to(PROJECT_ROOT)}...")
    try:
        df = pd.read_parquet(input_parquet)
        print(f"    -> Cargadas: {len(df):,} filas × {df.shape[1]} columnas")

        output_jsonl = STAGING_DIR / "dataset_consolidado.jsonl"
        print(f"  Guardando en JSON Lines en staging...")
        df.to_json(output_jsonl, orient="records", lines=True)

        print(f"\n[OK] Datos listos para entrenamiento. Creado: {output_jsonl.relative_to(PROJECT_ROOT)}")
        print(f"     ({output_jsonl.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as exc:
        print(f"\n[ERROR] Error al preparar datos de entrenamiento: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 01 COMPLETADA")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
