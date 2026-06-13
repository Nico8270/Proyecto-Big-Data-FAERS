"""
src/hadoop/training/04_evaluation.py
============================================
ETAPA 04: EVALUACIÓN DE RESULTADOS Y DESEMPEÑO DEL MODELO

Carga el modelo predictivo (modelo_severidad.joblib) y el conjunto de prueba
limpio desde data/staging/training_features/features_test.jsonl.

CORRECCIÓN DE DATA LEAKAGE — garantías de evaluación limpia:
  - X_test e y_test se leen directamente de features_test.jsonl.
    Ese archivo fue generado en la etapa 02 ANTES de que ocurriera cualquier
    resampling: no contiene muestras sintéticas de SMOTE/ADASYN.
  - NO se aplica ningún resampling sobre X_test ni y_test.
  - NO se vuelve a hacer train_test_split; el split ya fue fijo en la etapa 02.
  - El modelo se evalúa sobre datos reales, no sinteticos → métricas honestas.

Salidas:
  outputs/balance_reports/model_evaluation_metrics.json
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "staging" / "training_features"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "balance_reports"

# Debe coincidir con las columnas producidas en la etapa 02 y 03
FEATURE_COLS = ["drug_encoded", "pt_encoded", "sex_encoded", "age"]


def main():
    print("\n" + "=" * 60)
    print("  ETAPA 04: EVALUACIÓN DEL MODELO (TEST SET LIMPIO)")
    print("=" * 60)

    # Validar que el modelo existe
    model_path = ASSETS_DIR / "modelo_severidad.joblib"
    if not model_path.exists():
        print(f"\n[ERROR] Modelo entrenado no encontrado: {model_path}")
        sys.exit(1)

    # Cargar el conjunto de prueba (NO modificado, NO contiene sintéticos SMOTE)
    test_file = FEATURES_DIR / "features_test.jsonl"
    if not test_file.exists():
        print(f"\n[ERROR] Archivo de prueba no encontrado: {test_file}")
        print("        Asegúrese de haber ejecutado la etapa 02 (feature engineering).")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Cargando modelo desde  : {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Cargando X_test desde  : {test_file.relative_to(PROJECT_ROOT)}")
    t0 = time.time()

    try:
        # --- Step 1: Cargar modelo y datos de prueba ---
        clf    = joblib.load(model_path)
        df_test = pd.read_json(test_file, orient="records", lines=True)
        print(f"    -> {len(df_test):,} muestras de prueba cargadas.")

        # --- Step 2: Extraer X_test e y_test — SIN resampling de ningún tipo ---
        # X_test: sólo las features numéricas usadas en entrenamiento.
        # y_test: etiquetas reales del conjunto de prueba (no sintéticas).
        X_test = df_test[FEATURE_COLS].copy()
        y_test = df_test["severity_level"].astype(int)

        # --- Reescalar etiquetas de [1,2,3,4,5] a [0,1,2,3,4] para comparación ---
        # El modelo fue entrenado con etiquetas en [0,1,2,3,4], así que las predicciones
        # estarán en ese rango. Necesitamos hacer el mismo shift en y_test.
        class_labels = sorted(y_test.unique().tolist())
        label_shift  = min(class_labels)
        
        if label_shift != 0:
            y_test_shifted = y_test - label_shift
        else:
            y_test_shifted = y_test
        
        print(f"\n  Distribución real de clases en X_test (datos originales):")
        for cls, cnt in sorted(y_test.value_counts().items()):
            print(f"    Clase {cls}: {cnt:,} muestras")

        # --- Step 3: Predicciones sobre X_test (intacto) ---
        # El modelo nunca vio estos datos durante entrenamiento ni SMOTE.
        y_pred = clf.predict(X_test)

        # --- Step 4: Calcular métricas de clasificación ---
        accuracy = accuracy_score(y_test_shifted, y_pred)
        conf_mat = confusion_matrix(y_test_shifted, y_pred)
        clf_rep  = classification_report(y_test_shifted, y_pred, output_dict=True)

        dur = time.time() - t0

        # Imprimir métricas en consola
        print("\n  [MÉTRICAS GLOBAL] Exactitud del Modelo (Accuracy): {:.2%}".format(accuracy))
        print("\n  [REPORTE DE CLASIFICACIÓN POR NIVEL DE SEVERIDAD]")
        print("  " + "-" * 56)
        print("  {:<12} {:<10} {:<10} {:<10} {:<10}".format(
            "Gravedad", "Precisión", "Recall", "F1-Score", "Muestras"
        ))
        print("  " + "-" * 56)

        gravedades = {1: "Leve", 2: "Moderado", 3: "Importante", 4: "Grave", 5: "Muy Grave"}
        for shifted_k in sorted(clf_rep.keys()):
            if shifted_k == "accuracy" or not shifted_k.isdigit():
                continue
            shifted_k_int = int(shifted_k)
            orig_k = shifted_k_int + label_shift
            v = gravedades.get(orig_k, f"Clase {orig_k}")
            metrics = clf_rep.get(str(shifted_k_int), {})
            if metrics:
                print("  {:<12} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>10,}".format(
                    v,
                    metrics.get("precision", 0) * 100,
                    metrics.get("recall",    0) * 100,
                    metrics.get("f1-score",  0) * 100,
                    int(metrics.get("support", 0)),
                ))
        print("  " + "-" * 56)

        # Imprimir matriz de confusión
        print("\n  [MATRIZ DE CONFUSIÓN (FILAS: REALES, COLUMNAS: PREDICHAS)]")
        print("  " + "-" * 56)
        for i, row in enumerate(conf_mat):
            row_str = " ".join("{:>8,}".format(val) for val in row)
            orig_class = i + label_shift
            print(f"  Clase {orig_class} real    | {row_str}")
        print("  " + "-" * 56)
        print(f"\n  Tiempo de evaluación: {dur:.1f} segundos")

        # --- Step 5: Guardar métricas en JSON ---
        metrics_json = REPORTS_DIR / "model_evaluation_metrics.json"
        
        # Revertir el shift en el classification_report para guardar con etiquetas originales
        clf_rep_original = {}
        for key, value in clf_rep.items():
            if isinstance(value, dict) and "precision" in value:
                # Es una métrica por clase (no "accuracy", "macro avg", etc)
                try:
                    shifted_cls = int(key)
                    orig_cls = shifted_cls + label_shift
                    clf_rep_original[str(orig_cls)] = value
                except (ValueError, TypeError):
                    clf_rep_original[key] = value
            else:
                clf_rep_original[key] = value
        
        eval_data = {
            "accuracy":               float(accuracy),
            "classification_report":  clf_rep_original,
            "confusion_matrix":       conf_mat.tolist(),
            "n_test_samples":         int(len(y_test)),
            "label_shift_applied":    int(label_shift),
            "original_class_range":   class_labels,
            "model_class_range":      [c - label_shift for c in class_labels],
            "model_type":             "Random Forest Multiclass (depth=16, SMOTE on train only)",
            "data_leakage_free":      True,
            "nota":                   (
                "X_test proviene de train_test_split aplicado ANTES de SMOTE. "
                "No contiene muestras sintéticas. Las métricas son honestas. "
                "Las clases fueron desplazadas durante entrenamiento para compatibilidad con XGBoost."
            ),
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        print(f"\n[OK] Evaluación completada. Métricas en: {metrics_json.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"\n[ERROR] Excepción al evaluar el modelo: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 04 COMPLETADA")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
