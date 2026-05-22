"""
src/cleaning/s01_limpiar_demo.py
=================================
Limpieza específica de la tabla DEMO (demografía de pacientes) usando PySpark.

Lee de  : data/raw/faers/DEMO25Q4.txt
Escribe : data/clean_data/DEMO_clean.parquet (directorio Parquet de PySpark)
Log     : src/cleaning/logs/s01_demo_YYYY-MM-DD.txt
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

# Insertar el directorio actual para importar las reglas compartidas
sys.path.insert(0, str(Path(__file__).parent))
from rules_cleaning import RAW_DATA, CLEAN_DATA, FAERS_SEP, FAERS_ENCODING, RULES_DEMO


def obtener_spark_session() -> SparkSession:
    """Inicializa y devuelve una sesión de Spark local robusta para Windows."""
    return SparkSession.builder \
        .master("local[*]") \
        .appName("FAERS_Limpieza_DEMO") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()


def cargar_demo(spark: SparkSession) -> tuple:
    """Carga el DEMO crudo usando PySpark manejando rutas en Windows."""
    # Convertir la ruta a un URI formal para evitar problemas con espacios y OneDrive
    raw_path = str(Path(RAW_DATA / "DEMO25Q4.txt").resolve().as_posix())
    
    # Cargar archivo delimitado por $
    encoding = FAERS_ENCODING
    if encoding.lower() == "latin-1":
        encoding = "ISO-8859-1"
        
    df = spark.read \
        .option("header", "true") \
        .option("sep", FAERS_SEP) \
        .option("encoding", encoding) \
        .option("inferSchema", "true") \
        .csv(raw_path)
    
    # Normalizar los nombres de columnas (strip y lower)
    for col in df.columns:
        df = df.withColumnRenamed(col, col.strip().lower())
        
    return df


def limpiar_demo(df, spark: SparkSession) -> tuple:
    """Aplica todas las reglas de limpieza sobre DEMO usando PySpark."""
    log = {}
    
    # Filas iniciales (ejecución eager para registro de auditoría)
    filas_iniciales = df.count()
    log["filas_iniciales"] = filas_iniciales
    print(f"  Cargado: {filas_iniciales:,} filas × {len(df.columns)} columnas")
    
    # ── 1. Eliminar sin primaryid ──────────────────────────────────────────
    df_filtered = df.filter(F.col("primaryid").isNotNull() & (F.trim(F.col("primaryid")) != ""))
    filas_con_id = df_filtered.count()
    eliminados_sin_id = filas_iniciales - filas_con_id
    log["eliminados_sin_primaryid"] = eliminados_sin_id
    print(f"  Sin primaryid eliminados: {eliminados_sin_id:,}")
    
    # ── 2. Normalizar sexo ──────────────────────────────────────────────────
    sexo_default = RULES_DEMO["sexo_default"]
    sexo_validos = list(RULES_DEMO["sexo_validos"])
    
    # Si es nulo, vacío o no es una opción válida, se asigna el valor por defecto
    df_filtered = df_filtered.withColumn(
        "sex",
        F.when(
            F.col("sex").isNull() | (F.trim(F.col("sex")) == "") | ~F.trim(F.col("sex")).isin(sexo_validos),
            F.lit(sexo_default)
        ).otherwise(F.trim(F.col("sex")))
    )
    
    normalizados_sex = df_filtered.filter(F.col("sex") == sexo_default).count()
    log["sex_normalizados"] = normalizados_sex
    print(f"  Sexo normalizado: {normalizados_sex:,}")
    
    # ── 3. Limpiar edad ────────────────────────────────────────────────────
    df_filtered = df_filtered.withColumn("age", F.col("age").cast(DoubleType()))
    
    edad_min = RULES_DEMO["edad_min"]
    edad_max = RULES_DEMO["edad_max"]
    
    # Fuera de rango se marca como nulo
    df_filtered = df_filtered.withColumn(
        "age",
        F.when((F.col("age") >= edad_min) & (F.col("age") <= edad_max), F.col("age"))
        .otherwise(F.lit(None))
    )
    
    # Calcular la mediana de las edades válidas
    mediana_res = df_filtered.stat.approxQuantile("age", [0.5], 0.01)
    mediana_edad = mediana_res[0] if (mediana_res and mediana_res[0] is not None) else 45.0
    
    edad_fuera_rango = df_filtered.filter(F.col("age").isNull()).count()
    log["edad_fuera_rango"] = edad_fuera_rango
    log["edad_imputada_mediana"] = edad_fuera_rango
    
    # Imputar la mediana
    df_filtered = df_filtered.fillna({"age": mediana_edad})
    log["edad_final_nulos"] = df_filtered.filter(F.col("age").isNull()).count()
    print(f"  Edad fuera de rango / imputada (mediana={mediana_edad:.1f}): {edad_fuera_rango:,}")
    
    # ── 4. Normalizar paises (mayúsculas + strip) ───────────────────────────
    if "reporter_country" in df_filtered.columns:
        df_filtered = df_filtered.withColumn(
            "reporter_country", 
            F.upper(F.trim(F.col("reporter_country")))
        )
        paises_no_std = df_filtered.filter(F.length(F.col("reporter_country")) != 2).count()
        log["paises_normalizados"] = paises_no_std
        print(f"  Países normalizados: {paises_no_std:,}")
    
    # ── 5. Eliminar columnas con > threshold de nulos ──────────────────────
    umbral = RULES_DEMO["drop_columns_threshold_pct"] / 100
    
    # Contar nulos por columna de forma distribuida y eficiente
    null_exprs = [F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1).otherwise(0)).alias(c) for c in df_filtered.columns]
    null_counts_row = df_filtered.select(*null_exprs).collect()[0]
    null_counts = null_counts_row.asDict()
    
    cols_eliminar = []
    for col, null_c in null_counts.items():
        pct_nulos = null_c / filas_con_id if filas_con_id > 0 else 0
        if pct_nulos > umbral:
            cols_eliminar.append(col)
            
    df_filtered = df_filtered.drop(*cols_eliminar)
    log["columnas_eliminadas"] = cols_eliminar
    print(f"  Columnas eliminadas (>{RULES_DEMO['drop_columns_threshold_pct']}% nulos): {cols_eliminar}")
    
    # ── 6. Eliminar duplicados completos ───────────────────────────────────
    filas_antes_dups = df_filtered.count()
    df_filtered = df_filtered.dropDuplicates()
    filas_finales = df_filtered.count()
    duplicados_eliminados = filas_antes_dups - filas_finales
    
    log["duplicados_eliminados"] = duplicados_eliminados
    print(f"  Duplicados eliminados: {duplicados_eliminados:,}")
    
    log["filas_finales"] = filas_finales
    log["filas_eliminadas"] = filas_iniciales - filas_finales
    
    return df_filtered, log


def guardar_log(log: dict, nombre_archivo: str):
    """Guarda el log de limpieza en src/cleaning/logs/."""
    LOG_DIR = Path(__file__).parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / nombre_archivo
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Limpieza DEMO (PySpark) — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 60 + "\n\n")
        for k, v in log.items():
            f.write(f"  {k:<40} {v}\n")
    print(f"  [LOG] {path}")


def main():
    print("\n  [s01] Limpieza DEMO (PySpark)")
    print("  " + "-" * 56)
    
    spark = None
    try:
        # 1. Obtener la sesión de Spark
        spark = obtener_spark_session()
        
        # 2. Cargar datos
        df = cargar_demo(spark)
        
        # 3. Limpiar datos
        df_clean, log = limpiar_demo(df, spark)
        
        # 4. Guardar en Parquet de forma segura en Windows
        # Para evitar problemas con HADOOP_HOME y winutils.exe al escribir Parquet en Windows,
        # convertimos el DataFrame final de Spark a Pandas y lo escribimos de forma nativa con PyArrow.
        print("  [Procesando] Convirtiendo DataFrame de Spark a Pandas para escritura nativa en Parquet...")
        df_pandas = df_clean.toPandas()
        
        out_path = CLEAN_DATA / "DEMO_clean.parquet"
        
        # Eliminar si ya existe un archivo o carpeta previa con ese nombre para evitar PermissionError en Windows
        if out_path.exists():
            import shutil
            if out_path.is_dir():
                shutil.rmtree(out_path)
            else:
                out_path.unlink()
        
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df_pandas)
        pq.write_table(table, out_path, compression="snappy")
        print(f"  [OK] Guardado Parquet en: {out_path}")
        
        # 5. Guardar log
        guardar_log(log, f"s01_demo_{datetime.now():%Y-%m-%d}.txt")
        
        # Resumen
        iniciales = log['filas_iniciales']
        finales = log['filas_finales']
        eliminadas = log['filas_eliminadas']
        pct_del = (eliminadas / iniciales * 100) if iniciales > 0 else 0
        print(f"\n  Resumen: {iniciales:,} -> {finales:,} (-{eliminadas:,} filas, -{pct_del:.1f}%)\n")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("  [ERROR CRÍTICO] Se produjo una excepción en el pipeline de PySpark:")
        print("!" * 80 + "\n")
        # Imprimir el traceback completo sin límites
        traceback.print_exc(file=sys.stderr)
        print("\n" + "!" * 80)
        sys.exit(1)
        
    finally:
        # Detener la sesión de Spark limpia
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
