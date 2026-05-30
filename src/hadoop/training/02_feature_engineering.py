"""
src/hadoop/training/02_feature_engineering.py
================================================
ETAPA 02: FEATURE ENGINEERING PARA MACHINE LEARNING

Lee el archivo JSONL preparado desde data/staging/training_input/dataset_consolidado.jsonl.
Extrae y codifica numéricamente las columnas necesarias para el clasificador de severidad:
- primaryid: Identificador
- sex: Sexo (codificado como float 0.0, 1.0, 0.5)
- age: Edad (normalizada)
- drugname: Nombre del fármaco (codificado numéricamente)
- pt: Término de reacción adversa (codificado numéricamente)
- severity_level: Variable objetivo (del 1 al 5)

Soporta carga híbrida inteligente: si los datos ya fueron codificados numéricamente antes
del balanceo (Option 4), los mapea directamente para evitar pérdidas.
Guarda los datos listos para entrenar en data/staging/training_features/features.jsonl.
"""

import sys
import shutil
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DIR    = PROJECT_ROOT / "data" / "staging" / "training_input"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "staging" / "training_features"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"

def main():
    print("\n" + "=" * 60)
    print("  ETAPA 02: FEATURE ENGINEERING PARA MACHINE LEARNING")
    print("=" * 60)

    input_file = INPUT_DIR / "dataset_consolidado.jsonl"
    if not input_file.exists():
        print(f"\n[ERROR] Archivo de entrada de preparación no encontrado: {input_file}")
        sys.exit(1)

    # Limpiar y recrear directorios
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
        except OSError as e:
            print(f"  [AVISO] No se pudo limpiar {OUTPUT_DIR}: {e}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Cargando datos preparados y estructurando variables categóricas...")
    try:
        df = pd.read_json(input_file, orient="records", lines=True)
        print(f"    -> Filas iniciales: {len(df):,}")

        # Validaciones de columnas críticas
        if "severity_level" not in df.columns:
            print("[ERROR] Falta la columna objetivo 'severity_level' en el dataset de entrenamiento.")
            sys.exit(1)

        # Mapeo y extracción híbrida inteligente
        df_final = pd.DataFrame()
        df_final["primaryid"] = df["primaryid"] if "primaryid" in df.columns else df.index
        df_final["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(45.0)
        df_final["severity_level"] = df["severity_level"].astype(int)

        # ── 1. Codificación de Sexo ─────────────────────────────────────────────
        if "sex_encoded" in df.columns:
            df_final["sex_encoded"] = df["sex_encoded"].fillna(0.5)
        else:
            sex_col = df["sex"].fillna("U").astype(str).str.upper().str.strip() if "sex" in df.columns else pd.Series(["U"] * len(df))
            sex_map = {"M": 1.0, "F": 0.0, "U": 0.5}
            df_final["sex_encoded"] = sex_col.map(sex_map).fillna(0.5)

        # ── 2. Codificación de Fármacos y Reacciones (Híbrida) ───────────────────
        
        # Mapear drug_encoded
        if "drug_encoded" in df.columns:
            df_final["drug_encoded"] = df["drug_encoded"].fillna(-1)
            df_final["drugname"] = df["drugname"] if "drugname" in df.columns else "UNKNOWN"
        else:
            print("  [INFO] Mapeando drugname crudo al vuelo...")
            drug_col = df["drugname"].fillna("UNKNOWN").astype(str).str.upper().str.strip() if "drugname" in df.columns else pd.Series(["UNKNOWN"] * len(df))
            drug_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_final["drug_encoded"] = drug_encoder.fit_transform(pd.DataFrame(drug_col)).ravel()
            df_final["drugname"] = drug_col
            joblib.dump(drug_encoder, ASSETS_DIR / "drug_encoder.joblib")

        # Mapear pt_encoded
        if "pt_encoded" in df.columns:
            df_final["pt_encoded"] = df["pt_encoded"].fillna(-1)
            df_final["pt"] = df["pt"] if "pt" in df.columns else "UNKNOWN"
        else:
            print("  [INFO] Mapeando pt crudo al vuelo...")
            pt_col = df["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip() if "pt" in df.columns else pd.Series(["UNKNOWN"] * len(df))
            pt_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_final["pt_encoded"] = pt_encoder.fit_transform(pd.DataFrame(pt_col)).ravel()
            df_final["pt"] = pt_col
            joblib.dump(pt_encoder, ASSETS_DIR / "pt_encoder.joblib")

        # Escribir el archivo final de features optimizado
        output_file = OUTPUT_DIR / "features.jsonl"
        print(f"  Escribiendo features para entrenamiento...")
        df_final.to_json(output_file, orient="records", lines=True)

        print(f"\n[OK] Feature Engineering completado exitosamente.")
        print(f"     Guardado: {output_file.relative_to(PROJECT_ROOT)} ({len(df_final):,} registros)")
    except Exception as exc:
        print(f"\n[ERROR] Error en Feature Engineering: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 02 COMPLETADA")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
