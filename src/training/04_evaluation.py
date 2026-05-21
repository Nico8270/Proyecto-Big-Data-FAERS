"""
Script 04: Evaluation de Resultados MapReduce
Evalúa y valida resultados del procesamiento distribuido
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mapreduce_results"
TRAINING_OUTPUT = PROJECT_ROOT / "training" / "outputs"


def evaluate_results():
    """Evalúa resultados de MapReduce."""
    print("=" * 60)
    print("ETAPA 04: EVALUATION")
    print("=" * 60)
    
    # Verificar resultados existentes
    result_files = list(OUTPUT_DIR.glob("*.csv")) if OUTPUT_DIR.exists() else []
    
    evaluation = {
        "results_found": len(result_files),
        "files": [f.name for f in result_files]
    }
    
    if not result_files:
        print("[WARN] No hay resultados MapReduce. Ejecutar primero los jobs.")
    else:
        print(f"[OK] {len(result_files)} archivos de resultados encontrados")
    
    with open(TRAINING_OUTPUT / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    print("=" * 60)
    print("ETAPA 04 COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_results()