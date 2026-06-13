"""
crear_sample.py
===============
Genera un subconjunto de datos de ejemplo coherente (1000 primaryid únicos)
a partir de los archivos crudos completos de FAERS, asegurando que cada
primaryid existe en TODAS las tablas relacionales (DEMO, DRUG, REAC, OUTC).

Esto permite ejecutar el pipeline completo sin descargar 340 MB del servidor FDA.

Uso:
    python crear_sample.py
"""

import sys
from pathlib import Path
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "faers"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"

DEMO_FILE = RAW_DATA_DIR / "DEMO25Q4.txt"
DRUG_FILE = RAW_DATA_DIR / "DRUG25Q4.txt"
REAC_FILE = RAW_DATA_DIR / "REAC25Q4.txt"
INDI_FILE = RAW_DATA_DIR / "INDI25Q4.txt"
OUTC_FILE = RAW_DATA_DIR / "OUTC25Q4.txt"
THER_FILE = RAW_DATA_DIR / "THER25Q4.txt"
RPSR_FILE = RAW_DATA_DIR / "RPSR25Q4.txt"

# Crear directorio de salida
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("\n" + "=" * 70)
    print("  GENERADOR DE SUBCONJUNTO DE DATOS DE EJEMPLO FAERS")
    print("=" * 70)
    
    # Verificar que los archivos crudos existan
    if not DEMO_FILE.exists():
        print(f"\n[ERROR] No encontrado: {DEMO_FILE}")
        print("Por favor, descarga los datos crudos de FDA en data/raw/faers/")
        print("Visita: https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/")
        sys.exit(1)
    
    print(f"\n[1/7] Cargando DEMO (demografía)...")
    df_demo = pd.read_csv(DEMO_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
    df_demo.columns = df_demo.columns.str.strip().str.lower()
    print(f"      Cargadas: {len(df_demo):,} filas")
    
    print(f"[2/7] Cargando DRUG (medicamentos)...")
    df_drug = pd.read_csv(DRUG_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
    df_drug.columns = df_drug.columns.str.strip().str.lower()
    print(f"      Cargadas: {len(df_drug):,} filas")
    
    print(f"[3/7] Cargando REAC (reacciones adversas)...")
    df_reac = pd.read_csv(REAC_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
    df_reac.columns = df_reac.columns.str.strip().str.lower()
    print(f"      Cargadas: {len(df_reac):,} filas")
    
    print(f"[4/7] Cargando OUTC (outcomes)...")
    df_outc = pd.read_csv(OUTC_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
    df_outc.columns = df_outc.columns.str.strip().str.lower()
    print(f"      Cargadas: {len(df_outc):,} filas")
    
    print(f"[5/7] Encontrando primaryid comunes a TODAS las tablas...")
    ids_demo = set(df_demo["primaryid"].dropna().unique())
    ids_drug = set(df_drug["primaryid"].dropna().unique())
    ids_reac = set(df_reac["primaryid"].dropna().unique())
    ids_outc = set(df_outc["primaryid"].dropna().unique())
    
    ids_common = ids_demo & ids_drug & ids_reac & ids_outc
    print(f"      IDs comunes encontrados: {len(ids_common):,}")
    
    # Seleccionar 1000 IDs aleatorios
    n_sample = min(1000, len(ids_common))
    ids_sample = list(random.sample(list(ids_common), n_sample))
    print(f"      Seleccionados para muestra: {n_sample:,}")
    
    print(f"\n[6/7] Filtrando tablas por IDs de muestra...")
    
    demo_sample = df_demo[df_demo["primaryid"].isin(ids_sample)].copy()
    drug_sample = df_drug[df_drug["primaryid"].isin(ids_sample)].copy()
    reac_sample = df_reac[df_reac["primaryid"].isin(ids_sample)].copy()
    outc_sample = df_outc[df_outc["primaryid"].isin(ids_sample)].copy()
    
    # Cargar opcionales (INDI, THER, RPSR) si existen
    indi_sample = None
    ther_sample = None
    rpsr_sample = None
    
    if INDI_FILE.exists():
        df_indi = pd.read_csv(INDI_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
        df_indi.columns = df_indi.columns.str.strip().str.lower()
        indi_sample = df_indi[df_indi["primaryid"].isin(ids_sample)].copy()
    
    if THER_FILE.exists():
        df_ther = pd.read_csv(THER_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
        df_ther.columns = df_ther.columns.str.strip().str.lower()
        ther_sample = df_ther[df_ther["primaryid"].isin(ids_sample)].copy()
    
    if RPSR_FILE.exists():
        df_rpsr = pd.read_csv(RPSR_FILE, sep="$", encoding="latin-1", on_bad_lines="skip", quoting=3)
        df_rpsr.columns = df_rpsr.columns.str.strip().str.lower()
        rpsr_sample = df_rpsr[df_rpsr["primaryid"].isin(ids_sample)].copy()
    
    print(f"\n[7/7] Guardando subconjunto en {SAMPLE_DATA_DIR}/...")
    
    # Guardar con separador $ para mantener consistencia con FAERS
    demo_sample.to_csv(SAMPLE_DATA_DIR / "DEMO25Q4.txt", sep="$", index=False, encoding="latin-1")
    drug_sample.to_csv(SAMPLE_DATA_DIR / "DRUG25Q4.txt", sep="$", index=False, encoding="latin-1")
    reac_sample.to_csv(SAMPLE_DATA_DIR / "REAC25Q4.txt", sep="$", index=False, encoding="latin-1")
    outc_sample.to_csv(SAMPLE_DATA_DIR / "OUTC25Q4.txt", sep="$", index=False, encoding="latin-1")
    
    if indi_sample is not None:
        indi_sample.to_csv(SAMPLE_DATA_DIR / "INDI25Q4.txt", sep="$", index=False, encoding="latin-1")
    
    if ther_sample is not None:
        ther_sample.to_csv(SAMPLE_DATA_DIR / "THER25Q4.txt", sep="$", index=False, encoding="latin-1")
    
    if rpsr_sample is not None:
        rpsr_sample.to_csv(SAMPLE_DATA_DIR / "RPSR25Q4.txt", sep="$", index=False, encoding="latin-1")
    
    print(f"\n  Muestra generada con éxito:")
    print(f"    • primaryid únicos:    {n_sample:,}")
    print(f"    • DEMO filas:          {len(demo_sample):,}")
    print(f"    • DRUG filas:          {len(drug_sample):,}")
    print(f"    • REAC filas:          {len(reac_sample):,}")
    print(f"    • OUTC filas:          {len(outc_sample):,}")
    if indi_sample is not None:
        print(f"    • INDI filas:          {len(indi_sample):,}")
    if ther_sample is not None:
        print(f"    • THER filas:          {len(ther_sample):,}")
    if rpsr_sample is not None:
        print(f"    • RPSR filas:          {len(rpsr_sample):,}")
    print(f"\n  Ubicación: {SAMPLE_DATA_DIR}/")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
