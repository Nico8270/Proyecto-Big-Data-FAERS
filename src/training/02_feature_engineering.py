"""
Script 02: Feature Engineering para MapReduce
Genera features optimizados para el procesamiento distribuido
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_OUTPUT = PROJECT_ROOT / "training" / "outputs"


def create_mapreduce_features():
    """Configura features para procesamiento en MapReduce."""
    print("=" * 60)
    print("ETAPA 02: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Features para detección de señales
    features = {
        "demographic": ["age", "age_grp", "sex", "reporter_country"],
        "drug": ["drugname", "prod_ai", "role_cod", "route"],
        "reaction": ["pt"],
        "temporal": ["event_dt"],
        "outcome": ["outc_cod"]
    }
    
    with open(TRAINING_OUTPUT / "features_config.json", "w") as f:
        json.dump(features, f, indent=2)
    
    # Definir joins para MapReduce
    joins = {
        "drug_reaction": {
            "tables": ["DRUG", "REAC"],
            "on": "primaryid",
            "filter": "role_cod = 'PS'"
        },
        "demo_outcome": {
            "tables": ["DEMO", "OUTC"],
            "on": "primaryid"
        }
    }
    
    with open(TRAINING_OUTPUT / "joins_config.json", "w") as f:
        json.dump(joins, f, indent=2)
    
    print("[OK] Features configurados para MapReduce")
    print("=" * 60)
    print("ETAPA 02 COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    create_mapreduce_features()