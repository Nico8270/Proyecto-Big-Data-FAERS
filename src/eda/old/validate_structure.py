"""
Script de validación: Verifica que todos los archivos necesarios existan
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

print("=" * 70)
print("  VALIDACIÓN DE ESTRUCTURA DEL PROYECTO")
print("=" * 70)

errors = []
warnings = []

# Verificar carpetas
print("\n[1/3] Verificando carpetas...")
for folder in ["data", "scripts", "outputs", "models"]:
    path = PROJECT_ROOT / folder
    if path.exists():
        print(f"  [OK] {folder}/ existe")
    else:
        errors.append(f"Carpeta faltante: {folder}/")
        print(f"  [ERROR] {folder}/ no existe")

# Verificar archivos FAERS en data/
print("\n[2/3] Verificando archivos FAERS en data/...")
expected_files = [
    "DEMO25Q4.txt", "DRUG25Q4.txt", "REAC25Q4.txt",
    "OUTC25Q4.txt", "INDI25Q4.txt", "RPSR25Q4.txt", "THER25Q4.txt"
]

for fname in expected_files:
    fpath = DATA_DIR / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / 1024**2
        print(f"  [OK] {fname} ({size_mb:.1f} MB)")
    else:
        errors.append(f"Archivo faltante: data/{fname}")
        print(f"  [ERROR] {fname} no encontrado")

# Verificar scripts
print("\n[3/3] Verificando scripts EDA...")
scripts_expected = [
    "00_run_all_eda.py", "01_load_and_overview.py", "02_demographic_analysis.py",
    "03_drug_analysis.py", "04_reaction_analysis.py", "05_outcome_analysis.py",
    "06_temporal_analysis.py", "07_cross_analysis.py", "08_summary_report.py",
    "config.py"
]

for script in scripts_expected:
    spath = SCRIPTS_DIR / script
    if spath.exists():
        print(f"  [OK] {script}")
    else:
        errors.append(f"Script faltante: {script}")
        print(f"  [ERROR] {script} no encontrado")

# Verificar archivos de documentación
print("\n[4/3] Verificando documentación...")
docs = ["informe.md", "README.md", "requirements.txt"]
for doc in docs:
    if (PROJECT_ROOT / doc).exists():
        print(f"  [OK] {doc}")
    else:
        warnings.append(f"Doc faltante: {doc}")
        print(f"  [ADVERTENCIA] {doc} no encontrado")

# Resumen final
print("\n" + "=" * 70)
print("  RESUMEN DE VALIDACIÓN")
print("=" * 70)

if errors:
    print(f"\n  ERRORES ({len(errors)}):")
    for e in errors:
        print(f"    x {e}")
else:
    print("\n  + Todos los archivos requeridos est%n presentes" % "á")  # Reemplaza el símbolo por texto plano

if warnings:
    print(f"\n  ADVERTENCIAS ({len(warnings)}):")
    for w in warnings:
        print(f"    ! {w}")

# Mostrar comando de ejecución
print("\n" + "=" * 70)
print("  PRÓXIMO PASO")
print("=" * 70)
print("\n  Para ejecutar el EDA completo, corre:")
print("    python scripts/00_run_all_eda.py")
print("\n  O ejecuta scripts individuales:")
print("    python scripts/01_load_and_overview.py")
print("    python scripts/02_demographic_analysis.py")
print("    ...")
print("\n  Los resultados se guardarán en: outputs/eda_results/")
print("=" * 70)

sys.exit(1 if errors else 0)
