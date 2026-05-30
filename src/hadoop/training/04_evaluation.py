"""
src/hadoop/training/04_evaluation.py
============================================
ETAPA 04: EVALUACIÓN DE RESULTADOS Y DESEMPEÑO DEL MODELO

Carga el modelo predictivo entrenado (modelo_severidad.joblib) y los features.
Realiza una evaluación rigurosa usando una partición de prueba (20%) no vista
para calcular métricas multiclase de Inteligencia Artificial:
- Accuracy global del modelo
- Precisión, Recall y F1-Score por cada nivel de severidad (del 1 al 5)
- Matriz de Confusión

Guarda las métricas resultantes en outputs/balance_reports/model_evaluation_metrics.json
para visualización de la plataforma.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "staging" / "training_features" / "features.jsonl"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "balance_reports"

def main():
    print("\n" + "=" * 60)
    print("  ETAPA 04: EVALUACIÓN DE RESULTADOS Y DESEMPEÑO DEL MODELO")
    print("=" * 60)

    # Validar archivos
    model_path = ASSETS_DIR / "modelo_severidad.joblib"
    if not model_path.exists():
        print(f"\n[ERROR] Modelo entrenado no encontrado: {model_path}")
        sys.exit(1)
    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Archivo de features no encontrado: {INPUT_FILE}")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Cargando modelo y calculando métricas de clasificación en validación...")
    t0 = time.time()

    try:
        # 1. Cargar modelo y datos
        clf = joblib.load(model_path)
        df = pd.read_json(INPUT_FILE, orient="records", lines=True)

        features = ["drug_encoded", "pt_encoded", "sex_encoded", "age"]
        X = df[features].copy()
        y = df["severity_level"].astype(int)

        # 2. Dividir train/test (80/20) para evaluar sobre datos no vistos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 3. Predicciones en el test set
        y_pred = clf.predict(X_test)
        
        # 4. Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        clf_rep  = classification_report(y_test, y_pred, output_dict=True)

        # Imprimir en consola de forma muy visual y estilizada
        print("\n  [MÉTRICAS GLOBAL] Exactitud del Modelo (Accuracy): {:.2%}".format(accuracy))
        print("\n  [REPORTE DE CLASIFICACIÓN POR NIVEL DE SEVERIDAD]")
        print("  " + "-" * 56)
        print("  {:<12} {:<10} {:<10} {:<10} {:<10}".format("Gravedad", "Precisión", "Recall", "F1-Score", "Muestras"))
        print("  " + "-" * 56)
        
        gravedades = {1: "Leve", 2: "Moderado", 3: "Importante", 4: "Grave", 5: "Muy Grave"}
        for k, v in gravedades.items():
            metrics = clf_rep.get(str(k), {})
            if metrics:
                print("  {:<12} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>10,}".format(
                    v,
                    metrics.get("precision", 0) * 100,
                    metrics.get("recall", 0) * 100,
                    metrics.get("f1-score", 0) * 100,
                    metrics.get("support", 0)
                ))
        print("  " + "-" * 56)

        # Imprimir matriz de confusión
        print("\n  [MATRIZ DE CONFUSIÓN (FILAS: REALES, COLUMNAS: PREDICHAS)]")
        print("  " + "-" * 56)
        for i, row in enumerate(conf_mat):
            row_str = " ".join("{:>8,}".format(val) for val in row)
            print(f"  Clase {i+1} real    | {row_str}")
        print("  " + "-" * 56)

        # Guardar en archivo JSON de evaluación del modelo
        metrics_json = REPORTS_DIR / "model_evaluation_metrics.json"
        eval_data = {
            "accuracy": float(accuracy),
            "classification_report": clf_rep,
            "confusion_matrix": conf_mat.tolist(),
            "n_total_evaluado": int(len(y_test)),
            "model_type": "Random Forest Multiclass (16 depth)"
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)
            
        print(f"\n[OK] Evaluación completada. Métricas guardadas en: {metrics_json.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"\n[ERROR] Excepcion al evaluar el modelo: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 04 COMPLETADA")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
