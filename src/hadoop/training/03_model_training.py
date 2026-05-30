"""
src/hadoop/training/03_model_training.py
============================================
ETAPA 03: ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING

Carga las features de Machine Learning preparadas desde data/staging/training_features/features.jsonl.
Entrena un clasificador RandomForest robusto de clasificación multiclase (predictivo)
para aprender a clasificar la severidad de las reacciones adversas (niveles del 1 al 5).
Guarda el modelo predictivo final en outputs/model_assets/modelo_severidad.joblib.
"""

import sys
import shutil
import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "staging" / "training_features" / "features.jsonl"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"

def main():
    print("\n" + "=" * 60)
    print("  ETAPA 03: ENTRENAMIENTO DE IA (RANDOM FOREST MULTICLASS)")
    print("=" * 60)

    # Validar que existe el archivo de entrada
    if not INPUT_FILE.exists():
        print(f"\n[ERROR] Archivo de features para entrenamiento no encontrado: {INPUT_FILE}")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("  Cargando features y entrenando clasificador de Inteligencia Artificial...")
    t0 = time.time()

    try:
        # Cargar los datos de features
        df = pd.read_json(INPUT_FILE, orient="records", lines=True)
        print(f"    -> Cargados {len(df):,} registros balanceados para entrenamiento.")

        # Separar matrices X e y
        features = ["drug_encoded", "pt_encoded", "sex_encoded", "age"]
        X = df[features].copy()
        y = df["severity_level"].astype(int)

        print("\n  [Entrenamiento] Iniciando RandomForestClassifier (100 árboles)...")
        # Instanciar Random Forest Classifier optimizado para multiprocesamiento
        clf = RandomForestClassifier(
            n_estimators = 100,
            max_depth    = 16,
            random_state = 42,
            n_jobs       = -1, # Utiliza todos los núcleos de CPU del usuario en Windows
            class_weight = "balanced"
        )
        
        # Ajustar el modelo
        clf.fit(X, y)
        dur = time.time() - t0

        print(f"  [Entrenamiento] Ajuste completado con éxito en {dur:.1f} segundos.")

        # Guardar el modelo entrenado
        model_path = ASSETS_DIR / "modelo_severidad.joblib"
        print(f"  [Guardado] Serializando modelo predictivo en: {model_path.relative_to(PROJECT_ROOT)}")
        joblib.dump(clf, model_path, compress=3)

        print(f"\n[OK] Modelo entrenado y guardado con éxito.")
        
    except Exception as exc:
        print(f"\n[ERROR] Excepcion al entrenar el modelo de Machine Learning: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("  ETAPA 03 COMPLETADA")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
