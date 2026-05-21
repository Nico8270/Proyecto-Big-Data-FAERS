"""
Training Main - Orquestador del pipeline FAERS Big Data
Ejecuta las 5 etapas en orden para procesamiento distribuido
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

SCRIPTS = [
    ("01_data_preparation.py", "Preparación de Datos"),
    ("02_feature_engineering.py", "Feature Engineering"),
    ("03_model_training.py", "Model Training"),
    ("04_evaluation.py", "Evaluación"),
    ("05_inference.py", "Inferencia")
]


def run_script(script_name: str, stage_name: str):
    """Ejecuta un script de training."""
    script_path = PROJECT_ROOT / "src" / "training" / script_name
    
    print(f"\n[{stage_name}] Ejecutando {script_name}...")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    if result.returncode != 0:
        print(f"[ERROR] {script_name} falló")
        return False
    
    return True


def main():
    print("=" * 60)
    print("  TRAINING PIPELINE - FAERS BIG DATA")
    print("=" * 60)
    
    for i, (script, name) in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] {name}")
        if not run_script(script, name):
            print("\n[ERROR] Pipeline interrumpido")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print("=" * 60)
    print("\nSiguiente paso:")
    print("  docker-compose run --rm hadoop-faers python3 hadoop/run_faers_pipeline.py --local")


if __name__ == "__main__":
    main()