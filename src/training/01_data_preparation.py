"""
Script 01: Preparación de Datos para Hadoop/HDFS
Transforma datos FAERS al formato esperado por MapReduce
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_OUTPUT = PROJECT_ROOT / "training" / "outputs"

# Verificar que existe data dir
if not DATA_DIR.exists():
    print(f"[ERROR] Data directory not found: {DATA_DIR}")
    sys.exit(1)

def prepare_hdfs_input():
    """Prepara archivos para procesamiento MapReduce."""
    print("=" * 60)
    print("ETAPA 01: PREPARACIÓN HDFS")
    print("=" * 60)
    
    TRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Schema FAERS para MapReduce
    schema = {
        "DEMO": ["primaryid", "caseid", "caseversion", "i_f_code", "event_dt", 
                 "mfr_dt", "init_fda_dt", "fda_dt", "rept_cod", "auth_num",
                 "mfr_num", "mfr_sndr", "lit_ref", "age", "age_cod", "age_grp",
                 "sex", "e_sub", "wt", "wt_cod", "rept_dt", "to_mfr", 
                 "occp_cod", "reporter_country", "occr_country"],
        "DRUG": ["primaryid", "caseid", "drug_seq", "role_cod", "drugname",
                 "prod_ai", "val_vbm", "route", "dose_vbm", "cum_dose_chr",
                 "cum_dose_unit", "dechal", "rechal", "lot_num", "exp_dt",
                 "nda_num", "dose_amt", "dose_unit", "dose_form", "dose_freq"],
        "REAC": ["primaryid", "caseid", "pt", "drug_rec_act"]
    }
    
    # Guardar schema
    with open(TRAINING_OUTPUT / "faers_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # Verificar archivos
    files_ok = []
    for table in ["DEMO", "DRUG", "REAC", "OUTC", "INDI", "THER", "RPSR"]:
        filepath = DATA_DIR / f"{table}25Q4.txt"
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            files_ok.append((table, size_mb))
            print(f"[OK] {table}25Q4.txt: {size_mb:.1f} MB")
        else:
            print(f"[WARN] {table}25Q4.txt not found")
    
    # Metadata para configuración MapReduce
    config = {
        "input_format": "text",
        "delimiter": "$",
        "encoding": "latin-1",
        "files_loaded": len(files_ok)
    }
    
    with open(TRAINING_OUTPUT / "mapreduce_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] {len(files_ok)} archivos FAERS listos para procesamiento")
    print("=" * 60)
    print("ETAPA 01 COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    prepare_hdfs_input()