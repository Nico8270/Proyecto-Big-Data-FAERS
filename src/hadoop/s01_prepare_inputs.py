"""
src/hadoop/s01_prepare_inputs.py
==================================
Lee los archivos *.parquet limpios de data/clean_data/, los convierte
al formato JSON Lines (JSONL), añade la etiqueta de la tabla de origen
("table": "TABLA") y los guarda en la carpeta data/staging/hadoop_join_input/.

Normaliza la columna 'primaryid' a string de enteros de manera uniforme para
resolver la discrepancia de tipos de datos (str, int, float) entre las tablas,
garantizando una intersección exitosa y un MapReduce Join correcto.

Argumentos CLI opcionales:
  --sample N   Limitar cada tabla a los primeros N registros (modo prueba).
"""

import argparse
import os
import shutil
import re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA = PROJECT_ROOT / "data" / "clean_data"
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "hadoop_join_input"

TABLAS = {
    "DEMO": "DEMO_clean.parquet",
    "DRUG": "DRUG_clean.parquet",
    "REAC": "REAC_clean.parquet",
    "OUTC": "OUTC_clean.parquet",
    "INDI": "INDI_clean.parquet",
    "THER": "THER_clean.parquet",
    "RPSR": "RPSR_clean.parquet",
}

def normalizar_primaryid_series(series: pd.Series) -> pd.Series:
    """
    Normaliza de manera vectorial y ultra-rápida la columna primaryid
    a strings de enteros numéricos limpios.
    Ejemplos:
      12345.0      -> "12345"
      "12345"      -> "12345"
      "US-12345"   -> "12345"
      NaN          -> None
    """
    # Convertir a str, manejar nulos
    s = series.astype(str).str.strip()
    
    # Si viene con decimales de float (ej: 12345.0), remover el '.0'
    s = s.str.replace(r'\.0$', '', regex=True)
    
    # Remover cualquier caracter no numérico
    s = s.str.replace(r'[^0-9]', '', regex=True)
    
    # Reemplazar valores vacíos con None
    s = s.replace('', None)
    
    return s

def main(sample_size: int | None = None):
    modo = f"MUESTRA {sample_size:,} filas (coordinadas por intersección estricta)" if sample_size else "COMPLETO"
    print(f"\n  [s01] Preparando entradas para Hadoop MapReduce ({modo})...")
    print("  " + "-" * 56)

    # Asegurar que el directorio de staging existe
    if STAGING_DIR.exists():
        try:
            shutil.rmtree(STAGING_DIR)
        except OSError as e:
            print(f"  [AVISO] No se pudo limpiar {STAGING_DIR}: {e}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    archivos_procesados = 0
    ids_validos = None

    # Primero calculamos la intersección real de IDs entre las tablas en modo muestra
    if sample_size:
        print("  [Muestra] Calculando intersección de primaryid para garantizar resultados...")
        try:
            sets_of_ids = []
            
            # Cargar y normalizar IDs de las tablas
            for nombre, archivo in TABLAS.items():
                parquet_path = CLEAN_DATA / archivo
                if parquet_path.exists():
                    df_ids = pq.read_table(parquet_path, columns=["primaryid"]).to_pandas()
                    df_ids = df_ids.dropna(subset=["primaryid"])
                    
                    # Normalizar IDs de forma consistente
                    ids_norm = normalizar_primaryid_series(df_ids["primaryid"])
                    ids_norm = ids_norm.dropna()
                    
                    sets_of_ids.append(set(ids_norm.unique()))
            
            if sets_of_ids:
                ids_interseccion = set.intersection(*sets_of_ids)
                print(f"    -> IDs comunes en las 7 tablas: {len(ids_interseccion):,}")
                
                # Relajar a principales (DEMO, DRUG, REAC) si la intersección de las 7 es vacía
                if len(ids_interseccion) == 0:
                    print("    [INFO] Intersección de 7 tablas vacía. Relajando a principales: DEMO, DRUG y REAC...")
                    sets_principales = []
                    for nombre in ["DEMO", "DRUG", "REAC"]:
                        parquet_path = CLEAN_DATA / TABLAS[nombre]
                        if parquet_path.exists():
                            df_ids = pq.read_table(parquet_path, columns=["primaryid"]).to_pandas()
                            df_ids = df_ids.dropna(subset=["primaryid"])
                            ids_norm = normalizar_primaryid_series(df_ids["primaryid"]).dropna()
                            sets_principales.append(set(ids_norm.unique()))
                    
                    if sets_principales:
                        ids_interseccion = set.intersection(*sets_principales)
                        print(f"    -> IDs comunes en principales (DEMO, DRUG, REAC): {len(ids_interseccion):,}")
                
                # Seleccionar IDs de muestra
                if len(ids_interseccion) > 0:
                    ids_validos = set(list(ids_interseccion)[:sample_size])
                    print(f"    -> Seleccionados {len(ids_validos):,} IDs válidos para el filtrado de staging.")
                else:
                    print("    [AVISO] No se encontraron IDs comunes incluso en tablas principales.")
        except Exception as e:
            print(f"    [AVISO] Falló el cálculo avanzado de intersección de IDs: {e}")

    for nombre, archivo in TABLAS.items():
        parquet_path = CLEAN_DATA / archivo
        if not parquet_path.exists():
            print(f"  [AVISO] Archivo limpio no encontrado: {archivo}")
            continue

        print(f"  Procesando {nombre:<6} desde {archivo}...")
        try:
            # Leer el archivo parquet
            df = pq.read_table(parquet_path).to_pandas()

            if "primaryid" not in df.columns:
                print(f"  [ERROR] {nombre} no tiene la columna 'primaryid'. Omitiendo.")
                continue

            # Eliminar registros sin primaryid
            df = df.dropna(subset=["primaryid"])
            
            # Normalizar los IDs de la columna de manera consistente
            df["primaryid"] = normalizar_primaryid_series(df["primaryid"])
            df = df.dropna(subset=["primaryid"])

            # Aplicar filtro inteligente de IDs en modo muestra/prueba
            if sample_size and ids_validos is not None:
                df = df[df["primaryid"].isin(ids_validos)]
                print(f"    [Filtro] Registros coincidentes con la muestra: {len(df):,}")
            elif sample_size:
                # Fallback por si la intersección falló
                df = df.head(sample_size)
                print(f"    [MUESTRA] Limitado a {sample_size:,} filas.")

            # Tag de tabla de origen
            df["table"] = nombre

            # Guardar como JSON Lines (JSONL)
            jsonl_path = STAGING_DIR / f"{nombre.lower()}.jsonl"
            df.to_json(jsonl_path, orient="records", lines=True)

            print(f"    -> Creado: {jsonl_path.name} ({len(df):,} registros)")
            archivos_procesados += 1
        except Exception as exc:
            print(f"  [ERROR] Error procesando {nombre}: {exc}")

    if archivos_procesados > 0:
        print(f"\n  [OK] Preparacion de entradas completada. {archivos_procesados} archivos en: {STAGING_DIR}\n")
    else:
        print("\n  [ERROR] No se preparo ningun archivo de entrada. Verifique que existan los parquets limpios.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Limitar cada tabla a N filas (modo prueba)")
    args = parser.parse_args()
    main(sample_size=args.sample)
