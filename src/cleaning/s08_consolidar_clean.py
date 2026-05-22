"""
src/cleaning/s08_consolidar_clean.py
======================================
Une todas las tablas limpias en un solo dataset consolidado usando PySpark
para soportar el alto volumen de datos (evitando OutOfMemory en Pandas).

Entrada : data/clean_data/  -> archivos *_clean.parquet (s07 tablas)
Salida  : data/joined/dataset_consolidado.parquet
Log     : src/cleaning/logs/s08_consol_YYYY-MM-DD.txt
"""

import sys
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, LongType

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

def obtener_spark_session() -> SparkSession:
    return SparkSession.builder \
        .master("local[*]") \
        .appName("FAERS_Consolidacion") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()


def consolidar_con_spark(spark: SparkSession):
    dfs = {}
    print("\n  Cargando tablas...")
    
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
    
    # Comentar el guardado
    # consolidated_dir = CLEAN.parent / "joined"
    # consolidated_dir.mkdir(parents=True, exist_ok=True)
    # out = consolidated_dir / "dataset_consolidado.parquet"
    # print(f"\n  Guardando archivo consolidado (PySpark Parquet) en: {out}")
    # try:
    #     df_join.write.mode("overwrite").parquet(str(out.resolve()))
    # except Exception as e:
    #     if "winutils" in str(e).lower() or "java.io.ioexception" in str(e).lower():
    #         pass 
    #     else:
    #         raise e
    # print(f"  [OK] Guardado completado en formato de directorio Parquet.")
    
    return total_filas


def main():
    print("\n  [s08] Consolidar tablas limpias (PySpark)")
    print("  " + "-" * 56)

    spark = obtener_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        consolidar_con_spark(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
