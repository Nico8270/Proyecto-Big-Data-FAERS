"""
src/hadoop/training/05_inference.py
============================================
ETAPA 05: INFERENCIA Y ALERTA DE RIESGOS EN TIEMPO REAL

Carga el modelo predictivo entrenado (modelo_severidad.joblib) y los encoders.
Carga los casos clínicos del dataset consolidado crudo (203,791 pacientes).
Aplica el modelo para PREDECIR el nivel de severidad (1 al 5) de cada caso clínico
de manera personalizada.
Si el modelo predice una severidad Grave (4) o Muy Grave (5), dispara una alerta.
Guarda las alertas de alto riesgo resultantes en data/alerts/high_risk_alerts.json.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "data" / "clean_data" / "dataset_consolidado.parquet"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"
ALERTS_DIR   = PROJECT_ROOT / "data" / "alerts"

def main():
    print("\n" + "=" * 60)
    print("  ETAPA 05: INFERENCIA Y GENERACIÓN DE ALERTAS DE ALTO RIESGO")
    print("=" * 60)

    # Validar archivos requeridos
    model_path = ASSETS_DIR / "modelo_severidad.joblib"
    drug_enc_path = ASSETS_DIR / "drug_encoder.joblib"
    pt_enc_path = ASSETS_DIR / "pt_encoder.joblib"

    leaking_pt_path = ASSETS_DIR / "leaking_pt_terms.json"

    if not model_path.exists() or not drug_enc_path.exists() or not pt_enc_path.exists():
        print(f"\n[ERROR] Componentes del modelo no encontrados en: {ASSETS_DIR}")
        print("  Asegúrese de haber ejecutado con éxito las etapas 02 y 03 de entrenamiento.")
        sys.exit(1)

    # Cargar lista de PT terms enmascarados durante entrenamiento
    # (generada por 02_feature_engineering.py como parte de la corrección de leakage)
    if leaking_pt_path.exists():
        with open(leaking_pt_path, "r", encoding="utf-8") as _f:
            leaking_pt_set = set(json.load(_f))
        print(f"  [INFO] {len(leaking_pt_set)} PT terms de outcome-proxy cargados para enmascarar.")
    else:
        leaking_pt_set = set()
        print("  [AVISO] leaking_pt_terms.json no encontrado — inferencia sin PT masking.")

    if not INPUT_PARQUET.exists():
        print(f"\n[ERROR] Dataset consolidado para inferencia no encontrado: {INPUT_PARQUET}")
        sys.exit(1)

    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Cargando modelo entrenado y decodificadores categóricos...")
    t0 = time.time()

    try:
        clf = joblib.load(model_path)
        drug_encoder = joblib.load(drug_enc_path)
        pt_encoder = joblib.load(pt_enc_path)

        print(f"  Leyendo casos clínicos para inferencia: {INPUT_PARQUET.name}...")
        df = pd.read_parquet(INPUT_PARQUET)
        print(f"    -> Cargados: {len(df):,} pacientes para diagnóstico de riesgo.")

        # Preparar y normalizar los datos de entrada para la predicción
        df_inf = df.copy()
        
        # Homologar y limpiar columnas
        df_inf["drugname"] = df_inf["drugname"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        df_inf["pt"]       = df_inf["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        df_inf["sex"]      = df_inf["sex"].fillna("U").astype(str).str.upper().str.strip()
        df_inf["age"]      = pd.to_numeric(df_inf["age"], errors="coerce").fillna(45.0)

        # Codificar sexo (mapeo fijo — no depende del encoder)
        sex_map = {"M": 1.0, "F": 0.0, "U": 0.5}
        df_inf["sex_encoded"] = df_inf["sex"].map(sex_map).fillna(0.5)

        # Aplicar PT masking ANTES del encoder:
        # Los PT terms que son outcome-proxies deben recibir el centinela 'OUTCOME_PROXY'
        # para que el pt_encoder los trate de la misma manera que durante el entrenamiento.
        # Sin este paso, el encoder devolvería -1 (unknown) en lugar del código correcto
        # del centinela, causando predicciones inconsistentes.
        PT_SENTINEL = "OUTCOME_PROXY"
        if leaking_pt_set:
            pt_masked = df_inf["pt"].copy()
            pt_masked[pt_masked.isin(leaking_pt_set)] = PT_SENTINEL
            df_inf["pt_masked"] = pt_masked
            n_masked = (pt_masked == PT_SENTINEL).sum()
            print(f"  [PT masking] {n_masked:,} filas con PT outcome-proxy enmascarados.")
        else:
            df_inf["pt_masked"] = df_inf["pt"]

        # Aplicar encoders entrenados mapeando valores desconocidos a -1 de forma segura
        df_inf["drug_encoded"] = drug_encoder.transform(df_inf[["drugname"]]).ravel()
        df_inf["pt_encoded"]   = pt_encoder.transform(df_inf[["pt_masked"]]).ravel()

        # Separar matriz X de features
        features = ["drug_encoded", "pt_encoded", "sex_encoded", "age"]
        X = df_inf[features].copy()

        # --- Step 1: Ejecución de la Predicción con el Modelo de IA ---
        print("  [Inferencia] Calculando gravedades con el RandomForest de IA...")
        pred_severidades_shifted = clf.predict(X)
        
        # Revertir el shift de etiquetas: el modelo predice en [0,1,2,3,4]
        # pero necesitamos convertir a [1,2,3,4,5] para consistencia con el dataset original
        label_shift = 1  # Mismo shift que se aplicó en entrenamiento
        pred_severidades = pred_severidades_shifted + label_shift
        
        df_inf["predicted_severity"] = pred_severidades.astype(int)

        # Extraer probabilidades para la métrica de confianza (Score de riesgo)
        pred_probs = clf.predict_proba(X)
        # Tomar la probabilidad máxima de la clase predicha como confianza del modelo
        # pred_severidades_shifted está en [0,1,2,3,4], así que indexar directamente
        df_inf["confidence_score"] = [float(probs[pred]) for probs, pred in zip(pred_probs, pred_severidades_shifted)]

        # --- Step 2: Filtrado y Creación de Alertas de Alto Riesgo ---
        # Alertas críticas: Predicciones de severidad Grave (4) o Muy Grave (5)
        df_alertas = df_inf[df_inf["predicted_severity"].isin([4, 5])].copy()
        
        # Mapeo textual para el informe
        gravedades_text = {1: "Leve", 2: "Moderado", 3: "Importante", 4: "Grave", 5: "Muy Grave"}
        df_alertas["severity_label"] = df_alertas["predicted_severity"].map(gravedades_text)

        print(f"    -> Encontrados {len(df_alertas):,} reportes individuales con alerta de riesgo activo.")

        # Convertir a estructura JSON de alerta limpia
        alertas_list = []
        for idx, row in df_alertas.iterrows():
            alerta = {
                "primaryid": int(row.get("primaryid", 0)),
                "age": float(row.get("age", 45.0)),
                "sex": str(row.get("sex", "U")),
                "drugname": str(row.get("drugname", "DESCONOCIDO")),
                "pt": str(row.get("pt", "DESCONOCIDO")),
                "predicted_severity": int(row.get("predicted_severity")),
                "severity_label": str(row.get("severity_label")),
                "confidence_score": float(round(row.get("confidence_score", 0.0), 4)),
                "timestamp": datetime_now_str()
            }
            alertas_list.append(alerta)

        # Escribir el archivo final de alertas
        output_json = ALERTS_DIR / "high_risk_alerts.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(alertas_list, f, indent=2)

        dur = time.time() - t0
        print(f"\n[OK] Inferencia completada con éxito en {dur:.1f} segundos.")
        print(f"     Guardado: {output_json.relative_to(PROJECT_ROOT)} ({len(alertas_list):,} alertas de riesgo totales)")

    except Exception as exc:
        print(f"\n[ERROR] Error en Inferencia: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 05 COMPLETADA")
    print("=" * 60 + "\n")


def datetime_now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
