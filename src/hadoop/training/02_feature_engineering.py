"""
src/hadoop/training/02_feature_engineering.py
================================================
ETAPA 02: FEATURE ENGINEERING PARA MACHINE LEARNING

Lee el dataset consolidado desde data/staging/training_input/dataset_consolidado.jsonl.
Extrae y codifica numéricamente las columnas necesarias para el clasificador de severidad:
  - age         : Edad numérica (imputada con 45.0)
  - sex_encoded : Sexo (mapeo fijo M=1.0, F=0.0, U=0.5)
  - drug_encoded: Fármaco (OrdinalEncoder ajustado SOLO en X_train)
  - pt_encoded  : Reacción adversa (OrdinalEncoder ajustado SOLO en X_train,
                  con PTs que son outcome-proxies enmascarados como 'OUTCOME_PROXY')
  - severity_level: Variable objetivo (1-5)

CORRECCIÓN DE DATA LEAKAGE — dos capas:

  Capa 1 — Split antes de cualquier fit:
    1. Se define X e y desde el dataset original SIN balancear.
    2. train_test_split se aplica PRIMERO (test_size=0.20, random_state=42, stratify=y).
    3. Los OrdinalEncoder se ajustan (fit) SOLO sobre X_train; solo transforman X_test.
    4. SMOTE se aplicará en la etapa 03, SOLO sobre (X_train, y_train).
    5. X_test e y_test permanecen intactos para evaluación.

  Capa 2 — Eliminación de target leakage en pt_encoded:
    severity_level se deriva de outc_cod (FAERS OUTC table) de forma determinista.
    Ciertos MedDRA Preferred Terms (PT) están tan concentrados en un único valor de
    severity_level que revelan la etiqueta directamente (e.g. 'DEATH' -> sev=5).
    Estos PTs son OUTCOME PROXIES, no predictores independientes.
    Fix: antes de codificar pt, se mide la entropía de cada PT en X_train.
    Los PT con entropía < 0.5 se reemplazan por el centinela 'OUTCOME_PROXY',
    neutralizando la señal determinista mientras se preservan los términos legítimos.
    La lista de PTs enmascarados se guarda como leaking_pt_terms.json para reproducibilidad.

Salidas:
  data/staging/training_features/features_train.jsonl  ← para SMOTE + entrenamiento
  data/staging/training_features/features_test.jsonl   ← para evaluación (sin tocar)
  data/staging/training_features/features.jsonl        ← alias de train (compatibilidad)
  outputs/model_assets/leaking_pt_terms.json           ← lista de PTs enmascarados
  outputs/model_assets/pt_encoder.joblib               ← encoder ajustado solo en train
  outputs/model_assets/drug_encoder.joblib             ← encoder ajustado solo en train
"""

import json
import sys
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DIR    = PROJECT_ROOT / "data" / "staging" / "training_input"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "staging" / "training_features"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"

# Umbral de entropía: PTs con H < este valor se consideran outcome proxies
PT_ENTROPY_CUTOFF = 0.5
# Centinela para PTs enmascarados
PT_SENTINEL       = "OUTCOME_PROXY"
# Mínimo de muestras para calcular entropía fiablemente
PT_MIN_SAMPLES    = 3


def compute_pt_entropy(pt_series: pd.Series, sev_series: pd.Series) -> dict:
    """
    Calcula la entropía de Shannon de 'severity_level' dentro de cada PT term.
    Valores bajos (→ 0) indican PTs que determinan casi completamente la clase.
    Solo se evalúan PTs con al menos PT_MIN_SAMPLES muestras en X_train.
    """
    entropy_map = {}
    df_tmp = pd.DataFrame({"pt": pt_series.values, "sev": sev_series.values})
    for pt, grp in df_tmp.groupby("pt")["sev"]:
        if len(grp) >= PT_MIN_SAMPLES:
            vc = grp.value_counts(normalize=True)
            h = float(-(vc * np.log2(vc + 1e-10)).sum())
            entropy_map[pt] = h
    return entropy_map


def run_mi_audit(X: pd.DataFrame, y: pd.Series, label: str) -> pd.DataFrame:
    """
    Calcula mutual_info_classif para todas las features y devuelve un DataFrame
    ordenado. Imprime una tabla en consola.
    """
    mi_scores = mutual_info_classif(
        X, y, random_state=42,
        discrete_features=[False, False, True, True],  # age, sex continuous; drug, pt discrete
    )
    mi_df = pd.DataFrame({
        "feature": list(X.columns),
        "mutual_info": mi_scores,
    }).sort_values("mutual_info", ascending=False).reset_index(drop=True)

    print(f"\n  [{label}] Mutual Information vs severity_level:")
    print(f"  {'Feature':<20} {'MI Score':>10}  {'> 0.7 threshold':>18}")
    print("  " + "-" * 52)
    for _, row in mi_df.iterrows():
        flag = "  *** LEAKING ***" if row["mutual_info"] > 0.7 else ""
        print(f"  {row['feature']:<20} {row['mutual_info']:>10.4f}{flag}")
    return mi_df


def main():
    print("\n" + "=" * 60)
    print("  ETAPA 02: FEATURE ENGINEERING (SIN DATA LEAKAGE)")
    print("=" * 60)

    input_file = INPUT_DIR / "dataset_consolidado.jsonl"
    if not input_file.exists():
        print(f"\n[ERROR] Archivo de entrada no encontrado: {input_file}")
        sys.exit(1)

    # Limpiar y recrear directorios de salida
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
        except OSError as e:
            print(f"  [AVISO] No se pudo limpiar {OUTPUT_DIR}: {e}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Cargando dataset original (sin balancear)...")
    try:
        df = pd.read_json(input_file, orient="records", lines=True)
        print(f"    -> Filas cargadas: {len(df):,}")

        if "severity_level" not in df.columns:
            print("[ERROR] Falta la columna 'severity_level' en el dataset.")
            sys.exit(1)

        # --- Step 1: Construir X e y desde el dataset original ---
        # X incluye todas las features; 'severity_level' se extrae como y.
        # Ningún encoder ni SMOTE ha ocurrido todavía.

        age_col = pd.to_numeric(df["age"], errors="coerce").fillna(45.0)

        # Sexo: mapeo fijo, no depende del train (es conocimiento del dominio)
        if "sex_encoded" in df.columns:
            sex_col = pd.to_numeric(df["sex_encoded"], errors="coerce").fillna(0.5)
        else:
            raw_sex = (
                df["sex"].fillna("U").astype(str).str.upper().str.strip()
                if "sex" in df.columns else pd.Series(["U"] * len(df))
            )
            sex_col = raw_sex.map({"M": 1.0, "F": 0.0, "U": 0.5}).fillna(0.5)

        # Detectar si los datos llegaron precodificados (balanceo previo)
        # Nota: siempre preferimos trabajar con el texto crudo (pt, drugname)
        # para garantizar codificación consistente y evitar asunciones sobre pt_encoded/drug_encoded
        ya_codificado_drug = "drug_encoded" in df.columns and "drugname" not in df.columns
        ya_codificado_pt   = "pt_encoded" in df.columns and "pt" not in df.columns

        # Extraer columnas categóricas crudas para encodeado posterior
        # Siempre preferir el texto crudo si está disponible
        if "drugname" in df.columns:
            drug_col_raw = (
                df["drugname"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            )
        elif ya_codificado_drug:
            drug_col_raw = df["drug_encoded"].fillna(-1).astype(float)
        else:
            drug_col_raw = pd.Series(["UNKNOWN"] * len(df))

        # Para pt: siempre extraer el texto crudo para poder calcular entropía
        # (incluso si pt_encoded ya existe, necesitamos el string para auditoría)
        if "pt" in df.columns:
            pt_col_raw = df["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        elif ya_codificado_pt:
            # No hay texto; usamos el código numérico directamente sin enmascarar
            pt_col_raw = None
        else:
            pt_col_raw = pd.Series(["UNKNOWN"] * len(df))

        X_raw = pd.DataFrame({
            "age":          age_col.values,
            "sex_encoded":  sex_col.values,
            "drug_raw":     drug_col_raw.values if not ya_codificado_drug else drug_col_raw.values,
            "pt_raw":       pt_col_raw.values   if pt_col_raw is not None  else np.full(len(df), -1.0),
        })
        y = df["severity_level"].astype(int).reset_index(drop=True)

        # --- Step 2: train_test_split PRIMERO ---
        # Ningún encoder, scaler, ni SMOTE ha visto los datos todavía.
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"\n  [PASO 1] Split train/test completado:")
        print(f"    -> X_train : {len(X_train_raw):,} muestras")
        print(f"    -> X_test  : {len(X_test_raw):,} muestras (permanecerán intactas)")

        X_train = X_train_raw.copy().reset_index(drop=True)
        X_test  = X_test_raw.copy().reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)

        # --- Step 3: Enmascarar PT terms que son outcome proxies ---
        # Solo calculamos entropía en X_train para evitar que X_test influya en la
        # decisión de qué PTs enmascarar (también sería leakage).
        print(f"\n  [PASO 2] Auditoría de target leakage en pt_encoded...")

        if pt_col_raw is not None and not ya_codificado_pt:
            pt_train_text = X_train["pt_raw"].astype(str)
            pt_test_text  = X_test["pt_raw"].astype(str)

            entropy_map = compute_pt_entropy(pt_train_text, y_train)
            leaking_pts = sorted(
                pt for pt, h in entropy_map.items() if h < PT_ENTROPY_CUTOFF
            )
            safe_pts = [pt for pt in entropy_map if pt not in leaking_pts]

            print(f"    -> PT terms analizados (X_train)  : {len(entropy_map)}")
            print(f"    -> PT terms flagged (H < {PT_ENTROPY_CUTOFF})    : {len(leaking_pts)}")
            print(f"    -> PT terms seguros (preservados)  : {len(safe_pts)}")
            print(f"    -> Ejemplos de PTs enmascarados:")
            for pt in leaking_pts[:8]:
                h_val = entropy_map.get(pt, 0)
                print(f"         '{pt}' (H={h_val:.3f})")
            if len(leaking_pts) > 8:
                print(f"         ... y {len(leaking_pts)-8} más")

            # Aplicar máscara: leaking PTs -> sentinel
            pt_train_masked = pt_train_text.copy()
            pt_test_masked  = pt_test_text.copy()
            pt_train_masked[pt_train_masked.isin(leaking_pts)] = PT_SENTINEL
            pt_test_masked[pt_test_masked.isin(leaking_pts)]   = PT_SENTINEL

            # Guardar lista para reproducibilidad e inferencia
            leaking_json = ASSETS_DIR / "leaking_pt_terms.json"
            with open(leaking_json, "w", encoding="utf-8") as f:
                json.dump(leaking_pts, f, indent=2, ensure_ascii=False)
            print(f"    -> Lista guardada en: {leaking_json.relative_to(PROJECT_ROOT)}")
        else:
            pt_train_masked = X_train["pt_raw"].astype(str)
            pt_test_masked  = X_test["pt_raw"].astype(str)
            leaking_pts     = []
            print("    -> Texto crudo de PT no disponible; se omite auditoría de entropía.")

        # --- Step 4: Codificar drug y pt — fit SOLO sobre X_train ---
        print(f"\n  [PASO 3] Codificando features — fit SOLO en X_train...")

        # DRUG ENCODING
        # Si tenemos texto crudo (drugname), crear encoder
        # Si no, intentar usar valores precodificados
        if "drugname" in df.columns:
            drug_train_arr = np.array(X_train["drug_raw"].tolist(), dtype=str).reshape(-1, 1)
            drug_test_arr  = np.array(X_test["drug_raw"].tolist(),  dtype=str).reshape(-1, 1)
            drug_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train["drug_encoded"] = drug_encoder.fit_transform(drug_train_arr).ravel()
            X_test["drug_encoded"]  = drug_encoder.transform(drug_test_arr).ravel()
            joblib.dump(drug_encoder, ASSETS_DIR / "drug_encoder.joblib")
            print(f"    -> drug_encoder ajustado en X_train ({len(drug_encoder.categories_[0])} fármacos)")
        else:
            # Intentar interpretarlo como número precodificado
            try:
                X_train["drug_encoded"] = pd.to_numeric(X_train["drug_raw"], errors="raise").fillna(-1)
                X_test["drug_encoded"]  = pd.to_numeric(X_test["drug_raw"], errors="raise").fillna(-1)
                print(f"    -> drug_encoded usado directamente de valores precodificados")
            except (ValueError, TypeError):
                # Si no es un número, encodearlo como categoría
                drug_train_arr = np.array(X_train["drug_raw"].tolist(), dtype=str).reshape(-1, 1)
                drug_test_arr  = np.array(X_test["drug_raw"].tolist(),  dtype=str).reshape(-1, 1)
                drug_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_train["drug_encoded"] = drug_encoder.fit_transform(drug_train_arr).ravel()
                X_test["drug_encoded"]  = drug_encoder.transform(drug_test_arr).ravel()
                joblib.dump(drug_encoder, ASSETS_DIR / "drug_encoder.joblib")
                print(f"    -> drug_encoder creado para valores de texto ({len(drug_encoder.categories_[0])} categorías)")

        # PT ENCODING
        # Si tenemos texto crudo (pt), crear encoder
        # Si no, intentar usar valores precodificados
        if "pt" in df.columns:
            pt_train_arr = np.array(pt_train_masked.tolist(), dtype=str).reshape(-1, 1)
            pt_test_arr  = np.array(pt_test_masked.tolist(),  dtype=str).reshape(-1, 1)
            pt_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train["pt_encoded"] = pt_encoder.fit_transform(pt_train_arr).ravel()
            X_test["pt_encoded"]  = pt_encoder.transform(pt_test_arr).ravel()
            joblib.dump(pt_encoder, ASSETS_DIR / "pt_encoder.joblib")
            n_cats = len(pt_encoder.categories_[0])
            print(f"    -> pt_encoder ajustado en X_train ({n_cats} categorías, incl. centinela)")
        else:
            # Intentar interpretarlo como número precodificado
            try:
                X_train["pt_encoded"] = pd.to_numeric(X_train["pt_raw"], errors="raise").fillna(-1)
                X_test["pt_encoded"]  = pd.to_numeric(X_test["pt_raw"], errors="raise").fillna(-1)
                print(f"    -> pt_encoded usado directamente de valores precodificados")
            except (ValueError, TypeError):
                # Si no es un número, encodearlo como categoría
                pt_train_arr = np.array(X_train["pt_raw"].tolist(), dtype=str).reshape(-1, 1)
                pt_test_arr  = np.array(X_test["pt_raw"].tolist(),  dtype=str).reshape(-1, 1)
                pt_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_train["pt_encoded"] = pt_encoder.fit_transform(pt_train_arr).ravel()
                X_test["pt_encoded"]  = pt_encoder.transform(pt_test_arr).ravel()
                joblib.dump(pt_encoder, ASSETS_DIR / "pt_encoder.joblib")
                n_cats = len(pt_encoder.categories_[0])
                print(f"    -> pt_encoder creado para valores de texto ({n_cats} categorías)")

        # Conservar solo las columnas finales de features
        FINAL_COLS = ["age", "sex_encoded", "drug_encoded", "pt_encoded"]
        X_train_final = X_train[FINAL_COLS].copy()
        X_test_final  = X_test[FINAL_COLS].copy()

        # --- Step 5: MI audit ANTES y DESPUÉS del enmascaramiento ---
        print(f"\n  [PASO 4] Auditoría de Mutual Information (solo X_train)...")

        # MI con pt original (sin máscara) — solo para reportar delta
        X_train_unmasked = X_train_final.copy()
        if not ya_codificado_pt:
            pt_orig_arr = np.array(X_train["pt_raw"].tolist(), dtype=str).reshape(-1, 1)
            enc_tmp = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_unmasked["pt_encoded"] = enc_tmp.fit_transform(pt_orig_arr).ravel()
        mi_before = run_mi_audit(X_train_unmasked, y_train, "BEFORE PT masking")

        mi_after  = run_mi_audit(X_train_final, y_train, "AFTER PT masking")

        pt_mi_before = mi_before[mi_before["feature"] == "pt_encoded"]["mutual_info"].values[0]
        pt_mi_after  = mi_after[mi_after["feature"]   == "pt_encoded"]["mutual_info"].values[0]
        reduction    = (pt_mi_before - pt_mi_after) / (pt_mi_before + 1e-10) * 100

        print(f"\n  [RESUMEN DE LEAKAGE]")
        print(f"    pt_encoded MI antes : {pt_mi_before:.4f}")
        print(f"    pt_encoded MI después: {pt_mi_after:.4f}")
        print(f"    Reducción           : {reduction:.1f}%")
        if pt_mi_after <= 0.7:
            print("    [OK] pt_encoded está por debajo del umbral de leakage (0.7).")
        else:
            print("    [WARN] pt_encoded aún supera el umbral — se necesita más enmascaramiento.")

        # --- Step 6: Guardar splits ---
        train_df = X_train_final.copy()
        train_df["severity_level"] = y_train.values

        test_df = X_test_final.copy()
        test_df["severity_level"] = y_test.values

        train_file = OUTPUT_DIR / "features_train.jsonl"
        test_file  = OUTPUT_DIR / "features_test.jsonl"

        print(f"\n  Guardando features_train.jsonl ({len(train_df):,} registros)...")
        train_df.to_json(train_file, orient="records", lines=True)

        print(f"  Guardando features_test.jsonl  ({len(test_df):,} registros — NO se modificarán)...")
        test_df.to_json(test_file, orient="records", lines=True)

        # Alias para compatibilidad con scripts externos
        train_df.to_json(OUTPUT_DIR / "features.jsonl", orient="records", lines=True)

        print(f"\n[OK] Feature Engineering completado SIN data leakage.")
        print(f"     Train : {train_file.relative_to(PROJECT_ROOT)}")
        print(f"     Test  : {test_file.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"\n[ERROR] Error en Feature Engineering: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 02 COMPLETADA")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
