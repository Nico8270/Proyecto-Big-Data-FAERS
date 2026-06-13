"""
src/cleaning/s08_consolidar_clean.py
======================================
Une todas las tablas limpias en un solo dataset consolidado.
Usa PySpark de manera prioritaria, pero cuenta con un fallback robusto en Pandas/PyArrow
si Spark o Java no están configurados localmente.

Entrada : data/clean_data/  -> archivos *_clean.parquet (s07 tablas)
Salida  : data/joined/dataset_consolidado.parquet
Log     : src/cleaning/logs/s08_consol_YYYY-MM-DD.txt
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import CLEAN_DATA

LOG_DIR  = Path(__file__).parent / "logs"
CLEAN    = CLEAN_DATA

TABLAS = {
    "DEMO":  "DEMO_clean.parquet",
    "DRUG":  "DRUG_clean.parquet",
    "REAC":  "REAC_clean.parquet",
    "OUTC":  "OUTC_clean.parquet",
    "INDI":  "INDI_clean.parquet",
    "THER":  "THER_clean.parquet",
    "RPSR":  "RPSR_clean.parquet",
}

# Intentar importar PySpark de forma segura
try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.types import ArrayType, LongType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def obtener_spark_session() -> SparkSession:
    if not PYSPARK_AVAILABLE:
        raise RuntimeError("PySpark no está disponible.")
    return SparkSession.builder \
        .master("local[*]") \
        .appName("FAERS_Consolidacion") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()


def consolidar_con_pandas():
    """Fallback en Pandas/PyArrow para realizar el join y consolidación si Spark/Java no están configurados."""
    print("  [FALLBACK] Usando Pandas/PyArrow para realizar la consolidación de tablas...")
    import pandas as pd

    dfs = {}
    tablas_requeridas = ["DEMO", "DRUG", "REAC", "INDI", "OUTC"]

    for nombre in tablas_requeridas:
        archivo = TABLAS[nombre]
        path = CLEAN / archivo
        if not path.exists():
            print(f"  [AVISO] No encontrado: {path}")
            continue

        # Cargar con pandas/pyarrow
        df = pd.read_parquet(path)

        # Homologar nombre de primaryid
        join_col_temp = "primaryid"
        if "PRIMARYID" in df.columns and join_col_temp not in df.columns:
            df = df.rename(columns={"PRIMARYID": join_col_temp})

        # Fuerza Bruta de Limpieza (Regex): eliminar todo lo que no sea dígito
        # Convertimos a string, quitamos caracteres no numéricos y luego a numérico
        df[join_col_temp] = df[join_col_temp].astype(str).str.replace(r"[^0-9]", "", regex=True)
        # Reemplazar vacíos con NaN y convertir a Int64 nativo de pandas (permite nulos de tipo entero)
        df[join_col_temp] = pd.to_numeric(df[join_col_temp], errors="coerce").astype("Int64")

        # Renombrar columnas para evitar sufijos y conflictos (excepto primaryid)
        if nombre != "DEMO":
            renames = {col: f"{col}_{nombre.lower()}" for col in df.columns if col.lower() != join_col_temp}
            df = df.rename(columns=renames)

        dfs[nombre] = df
        print(f"    {nombre:<6}: {len(df):>10,} filas cargadas")

    if "DEMO" not in dfs or "DRUG" not in dfs:
        print("  [ERROR] DEMO y DRUG son obligatorias para esta prueba.")
        return None

    print("\n  Aislamiento Inicial: Ejecutando INNER JOIN (DEMO + DRUG) por primaryid...")
    df_demo = dfs["DEMO"]
    df_drug = dfs["DRUG"]

    # Muestra Reducida (100 filas de DEMO)
    df_demo_sample = df_demo.head(100)

    # Reporte estricto
    print("\n  [REPORTE] IDs de DEMO (Muestra de 100 filas):")
    print(df_demo_sample[["primaryid"]].head(5).to_string(index=False))

    print("\n  [REPORTE] IDs de DRUG:")
    print(df_drug[["primaryid"]].head(5).to_string(index=False))

    # Join - Forzando INNER
    df_join = pd.merge(df_demo_sample, df_drug, on="primaryid", how="inner")

    # Conteo Final
    total_filas = len(df_join)
    print(f"\n  [CHECKPOINT] df_join.count() -> Registros coincidentes: {total_filas:,}")

    return total_filas


def consolidar_con_spark(spark: SparkSession):
    dfs = {}
    print("\n  Cargando tablas con Spark...")
    
    # Solo las 5 tablas requeridas para la Fase 2
    tablas_requeridas = ["DEMO", "DRUG", "REAC", "INDI", "OUTC"]
    
    for nombre in tablas_requeridas:
        archivo = TABLAS[nombre]
        path = CLEAN / archivo
        if not path.exists():
            print(f"  [AVISO] No encontrado: {path}")
            continue
        
        # Cargar dataframe
        df = spark.read.parquet(str(path.resolve().as_posix()))
        
        # Homologar nombre de primaryid
        join_col_temp = "primaryid"
        if "PRIMARYID" in df.columns and join_col_temp not in df.columns:
            df = df.withColumnRenamed("PRIMARYID", join_col_temp)
            
        # Fuerza Bruta de Limpieza (Regex): eliminar todo lo que no sea dígito
        df = df.withColumn(join_col_temp, F.regexp_replace(F.col(join_col_temp).cast("string"), "[^0-9]", "").cast(LongType()))
        
        # Renombrar columnas para evitar sufijos y conflictos (excepto primaryid)
        if nombre != "DEMO":
            for col_name in df.columns:
                if col_name.lower() != join_col_temp:
                    df = df.withColumnRenamed(col_name, f"{col_name}_{nombre.lower()}")
                    
        dfs[nombre] = df
        print(f"    {nombre:<6}: {df.count():>10,} filas cargadas")

    if "DEMO" not in dfs or "DRUG" not in dfs:
        print("  [ERROR] DEMO y DRUG son obligatorias para esta prueba.")
        return None

    print("\n  Aislamiento Inicial: Ejecutando INNER JOIN (DEMO + DRUG) por primaryid...")
    df_demo = dfs["DEMO"]
    df_drug = dfs["DRUG"]
    
    # Muestra Reducida
    df_demo_sample = df_demo.limit(100)
    
    # Reporte estricto
    print("\n  [REPORTE] IDs de DEMO (Muestra de 100 filas):")
    df_demo_sample.select("primaryid").show(5)
    
    print("\n  [REPORTE] IDs de DRUG:")
    df_drug.select("primaryid").show(5)

    # Join - Forzando INNER
    join_col = "primaryid"
    df_join = df_demo_sample.join(df_drug, on=join_col, how="inner")
    
    # Conteo Final
    total_filas = df_join.count()
    print(f"\n  [CHECKPOINT] df_join.count() -> Registros coincidentes: {total_filas:,}")
    
    return total_filas


def main():
    print("\n  [s08] Consolidar tablas limpias (PySpark/Pandas)")
    print("  " + "-" * 56)

    spark = None
    try:
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark no importable.")
            
        spark = obtener_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
        consolidar_con_spark(spark)
    except Exception as e:
        print(f"\n  [INFO] Spark/Java no disponible o falló con: {e}")
        try:
            consolidar_con_pandas()
        except Exception as e_pandas:
            print(f"\n  [ERROR CRÍTICO] Falló el fallback de consolidación: {e_pandas}")
            sys.exit(1)
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
