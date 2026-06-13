"""
src/hadoop/s03_postprocess_output.py
======================================
Lee las salidas de la ejecución de MapReduce (guardadas en data/staging/hadoop_join_output/),
las procesa, las carga en un pandas DataFrame y guarda el resultado final en
data/clean_data/dataset_consolidado.parquet.
Finalmente, realiza la limpieza de los directorios temporales de staging.
"""

import os
import shutil
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DATA   = PROJECT_ROOT / "data" / "clean_data"
INPUT_DIR    = PROJECT_ROOT / "data" / "staging" / "hadoop_join_input"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "staging" / "hadoop_join_output"
LOG_DIR      = PROJECT_ROOT / "src" / "cleaning" / "logs"

def postprocesar():
    print("\n  [s03] Postprocesando salidas de MapReduce...")
    print("  " + "-" * 56)

    if not OUTPUT_DIR.exists():
        print(f"  [ERROR] Directorio de salida no encontrado: {OUTPUT_DIR}")
        return False

    records = []
    
    # Recorrer todos los archivos de salida generados por mrjob
    for filepath in OUTPUT_DIR.iterdir():
        # Ignorar archivos ocultos o de metadatos de Hadoop (ej. _SUCCESS)
        if filepath.is_dir() or filepath.name.startswith("_") or filepath.name.startswith("."):
            continue
        
        print(f"  Leyendo archivo de salida: {filepath.name}...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        # Saltar líneas inválidas por seguridad
                        pass
        except Exception as exc:
            print(f"  [ERROR] Al leer {filepath.name}: {exc}")

    if not records:
        print("\n  [ERROR] No se encontraron registros en las salidas de MapReduce.")
        return False

    print(f"\n  Cargando {len(records):,} registros combinados en DataFrame...")
    df = pd.DataFrame(records)

    # ── Mapeo para generar la variable objetivo severity_level a partir de outc_cod ──
    # Mapeo oficial: DE=5 (Muerte), LT=4 (Grave), HO/CA=3 (Importante), DS/RI=2 (Moderado), OT=1 (Leve)
    col_outc = None
    for col in df.columns:
        if "outc_cod" in col:
            col_outc = col
            break
            
    if col_outc:
        mapeo_severidad = {
            "DE": 5, "LT": 4, "HO": 3, "CA": 3, "DS": 2, "RI": 2, "OT": 1
        }
        df["outc_cod_clean"] = df[col_outc].astype(str).str.strip().str.upper()
        df["severity_level"] = df["outc_cod_clean"].map(mapeo_severidad).fillna(1).astype(int)
        df = df.drop(columns=["outc_cod_clean"])
        print(f"  [OK] Columna objetivo 'severity_level' generada con éxito a partir de '{col_outc}'.")
    else:
        df["severity_level"] = 1
        print("  [AVISO] No se encontró columna de outcomes. Se generó 'severity_level' por defecto (1).")

    # ── Mapeo y codificación de variables categóricas críticas antes del balanceo ──
    # Esto asegura que sean tratadas como numéricas por SMOTE/ENN y se conserven en dataset.parquet
    try:
        from sklearn.preprocessing import OrdinalEncoder
        import joblib
        
        ASSETS_DIR = PROJECT_ROOT / "outputs" / "model_assets"
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        # Imputar y normalizar columnas críticas
        if "drugname" in df.columns:
            df["drugname"] = df["drugname"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            drug_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df["drug_encoded"] = drug_encoder.fit_transform(df[["drugname"]]).ravel()
            joblib.dump(drug_encoder, ASSETS_DIR / "drug_encoder.joblib")
            
        if "pt" in df.columns:
            df["pt"] = df["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            pt_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df["pt_encoded"] = pt_encoder.fit_transform(df[["pt"]]).ravel()
            joblib.dump(pt_encoder, ASSETS_DIR / "pt_encoder.joblib")

        if "sex" in df.columns:
            df["sex"] = df["sex"].fillna("U").astype(str).str.upper().str.strip()
            sex_map = {"M": 1.0, "F": 0.0, "U": 0.5}
            df["sex_encoded"] = df["sex"].map(sex_map).fillna(0.5)

        print(f"  [OK] Codificación categórica finalizada. Encoders guardados en: {ASSETS_DIR.relative_to(PROJECT_ROOT)}")
    except Exception as e_enc:
        print(f"  [AVISO] Falló la codificación categórica anticipada: {e_enc}")

    # Validar primaryid
    dup_pk = df["primaryid"].duplicated().sum()
    print(f"  primaryid únicos: {df['primaryid'].nunique():,}")
    print(f"  primaryid duplicados: {dup_pk:,}")

    # Guardar en parquet
    out_parquet = CLEAN_DATA / "dataset_consolidado.parquet"
    print(f"  Guardando archivo consolidado en Parquet: {out_parquet.name}...")
    try:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_parquet, compression="snappy")
        print(f"  [OK] Guardado exitosamente: {out_parquet.name} ({len(df):,} filas × {df.shape[1]} columnas)")
    except Exception as exc:
        print(f"  [ERROR] Al guardar archivo parquet consolidado: {exc}")
        return False

    # Guardar log de consolidación para mantener compatibilidad
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"s08_consol_{datetime.now():%Y-%m-%d}.txt"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Consolidación Hadoop MapReduce — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"  Consolidado final: {len(df):,} filas × {df.shape[1]} columnas\n")
            f.write(f"  primaryid únicos: {df['primaryid'].nunique():,}\n")
            f.write(f"  primaryid duplicados: {dup_pk:,}\n")
        print(f"  [LOG] {log_path}")
    except Exception as e:
        print(f"  [AVISO] No se pudo guardar el archivo de log: {e}")

    # Limpieza de directorios temporales
    print("\n  Limpiando directorios temporales de staging...")
    for path in (INPUT_DIR, OUTPUT_DIR):
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"    -> Eliminado: {path.relative_to(PROJECT_ROOT)}")
            except OSError as e:
                print(f"    [AVISO] No se pudo eliminar completamente {path.name}: {e}")

    return True

if __name__ == "__main__":
    postprocesar()
