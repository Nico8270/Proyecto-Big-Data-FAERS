"""
Script 03: Model Training para Big Data
Entrenamiento distribuido usando MapReduce y Hadoop
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_OUTPUT = PROJECT_ROOT / "training" / "outputs"


def train_mapreduce_models():
    """Configura jobs MapReduce para entrenamiento."""
    print("=" * 60)
    print("ETAPA 03: MODEL TRAINING (DISTRIBUIDO)")
    print("=" * 60)
    
    # Jobs MapReduce disponibles en el proyecto
    jobs = {
        "drug_reaction_pairs": {
            "script": "drug_reaction_pairs.py",
            "input": ["DRUG", "REAC"],
            "output": "top_drug_reaction_pairs"
        },
        "top_drugs": {
            "script": "top_drugs.py",
            "input": ["DRUG"],
            "output": "top_drugs"
        },
        "top_reactions": {
            "script": "top_reactions.py",
            "input": ["REAC"],
            "output": "top_reactions"
        },
        "count_by_country": {
            "script": "count_by_country.py",
            "input": ["DEMO"],
            "output": "reports_by_country"
        }
    }
    
    with open(TRAINING_OUTPUT / "training_jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)
    
    print("Jobs MapReduce configurados:")
    for name, job in jobs.items():
        print(f"  - {name}: {job['script']}")
    
    # Configuración para ejecutar con Hadoop
    hadoop_cmd = f"""
# Ejecutar jobs individualmente:
python src/hadoop/mapper/run_drug_reaction_pairs.py --local
python src/hadoop/mapper/run_top_drugs.py --local
python src/hadoop/mapper/run_top_reactions.py --local
python src/hadoop/mapper/run_count_by_country.py --local
    """.strip()
    
    with open(TRAINING_OUTPUT / "run_training.sh", "w") as f:
        f.write(hadoop_cmd)
    
    print("\n[OK] Configuración de entrenamiento distribuido guardada")
    print("=" * 60)
    print("ETAPA 03 COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    train_mapreduce_models()