"""
src/hadoop/training/training_main.py
======================================
ORQUESTRADOR PRINCIPAL DEL PIPELINE DE ENTRENAMIENTO HADOOP MAPREDUCE

# Pipeline: local batch processing, architected for horizontal scalability.
# To deploy on EMR: change runner to 'emr' in mrjob.conf and set up AWS credentials.

Ejecuta secuencialmente las 5 etapas del pipeline de entrenamiento Big Data:
1. Etapa 01: data_preparation.py (Crea JSONL desde Parquet consolidado)
2. Etapa 02: feature_engineering.py (Genera features optimizados)
3. Etapa 03: model_training.py (Ejecuta el mrjob MRModelTraining escalable)
4. Etapa 04: evaluation.py (Calcula PRR/Chi-cuadrado y detecta senales)
5. Etapa 05: inference.py (Genera alertas clinicas de alto riesgo)

Finalmente, realiza la limpieza de todos los directorios temporales de staging.
"""

import sys
import shutil
import subprocess
import time
from pathlib import Path

# Determinar PROJECT_ROOT absoluto y resolverlo
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAINING_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("01_data_preparation.py", "PREPARACION DE DATOS"),
    ("02_feature_engineering.py", "FEATURE ENGINEERING"),
    ("03_model_training.py", "MODEL TRAINING (MAPREDUCE)"),
    ("04_evaluation.py", "EVALUACION DEL MODELO (SENALES DE SEGURIDAD)"),
    ("05_inference.py", "INFERENCIA Y ALERTA DE RIESGOS")
]

def run_stage(script_name: str, stage_name: str) -> bool:
    """Ejecuta un script de etapa de entrenamiento como subproceso."""
    script_path = TRAINING_DIR / script_name
    print(f"\n>>> INICIANDO ETAPA: {stage_name} ({script_name})...")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    dur = time.time() - t0
    
    if result.returncode == 0:
        print(f">>> ETAPA {stage_name} completada con exito en {dur:.1f} segundos.")
        return True
    else:
        print(f"\n[ERROR] La etapa {stage_name} ha fallado con codigo {result.returncode}.")
        return False

def main():
    print("=" * 60)
    print("  ORQUESTRADOR DEL PIPELINE DE ENTRENAMIENTO HADOOP MAPREDUCE")
    print("=" * 60)
    
    t_start = time.time()

    # Ejecutar secuencialmente cada etapa
    for i, (script, name) in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] Progresando: {name}")
        if not run_stage(script, name):
            print("\n[FALLO] Pipeline de entrenamiento interrumpido debido a errores.")
            sys.exit(1)

    # Limpieza final de todos los directorios temporales de staging de training
    print("\n>>> Realizando limpieza de directorios de staging...")
    staging_paths = [
        PROJECT_ROOT / "data" / "staging" / "training_input",
        PROJECT_ROOT / "data" / "staging" / "training_features",
        PROJECT_ROOT / "data" / "staging" / "training_output"
    ]
    for path in staging_paths:
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"    -> Eliminado: {path.relative_to(PROJECT_ROOT)}")
            except OSError as e:
                print(f"    [AVISO] No se pudo eliminar completamente {path.name}: {e}")

    total_dur = time.time() - t_start
    print("\n" + "=" * 60)
    print("  PIPELINE DE ENTRENAMIENTO HADOOP MAPREDUCE COMPLETADO CON ÉXITO")
    print(f"  Duracion total: {total_dur:.1f} segundos")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
